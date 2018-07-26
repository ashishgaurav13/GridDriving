import sys, math
import numpy as np

import Box2D
from Box2D.b2 import edgeShape, circleShape
from Box2D.b2 import fixtureDef, polygonShape
from Box2D.b2 import revoluteJointDef, contactListener

import gym
from gym import spaces
from car_dynamics import Car
from gym.utils import colorize, seeding
from grid_utils import *

import pyglet
from pyglet import gl

# Initially created by Oleg Klimov.
# Licensed on the same terms as the rest of OpenAI Gym.
# Modified by Ashish Gaurav.

STATE_W, STATE_H = 256, 256
VIDEO_W, VIDEO_H = 900, 500
# WINDOW_W, WINDOW_H = VIDEO_W, VIDEO_H
WINDOW_W, WINDOW_H = 512, 512
GRID_COLS, GRID_ROWS = 8, 4
PROB_EDGE = 0.75
RANDOM_DELETIONS = 12
LATTICE_CONSTRAINTS = [(0, 0), (0, 0), (0, np.inf), (2, np.inf), (2, np.inf)]

PLAYFIELD   = 800        # Game over boundary
FPS         = 50
ZOOM        = 3          # Camera zoom
ZOOM_FOLLOW = False      # Set to False for fixed view (don't use zoom)
SCALE       = 1.0

LANE_WIDTH = 12
LANE_SEP = 1
EDGE_WIDTH = 100
TRAFFIC_LIGHT_R = LANE_WIDTH//3
TRAFFIC_LIGHT_R2 = LANE_WIDTH//4

ConvertRGB = lambda x: map(lambda y: y/255.0, x)
TRAFFIC_LIGHT_OUTER_COLOR = ConvertRGB((80, 126, 27))
TRAFFIC_LIGHT_INNER_COLOR = ConvertRGB((255, 255, 255))
ROAD_COLOR = [0.4, 0.4, 0.4]
LANE_SEP_COLOR = [0.6, 0.6, 0.6]

class FrictionDetector(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile: return
        # show a streak behind the object with color ROAD_COLOR
        tile.color = ROAD_COLOR[:]
        if not obj or "tiles" not in obj.__dict__: return
        if tile.boxtype == "lane-sep": return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)

class CarGridDriving(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        # action space has steer (-1, 0, 1), gas (0, 1), brake (0, 1)
        self.action_space = spaces.Box( 
            low=np.array([-1,0,0]), 
            high=np.array([+1,+1,+1])
        )
        # produce an observation space of rows x cols x 3
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(STATE_H, STATE_W, 3), 
            dtype=np.uint8
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        self.off_params = (-WINDOW_H//4, -WINDOW_W//2.5)
        lattice = construct_lattice(GRID_ROWS, GRID_COLS, PROB_EDGE,
            LATTICE_CONSTRAINTS, RANDOM_DELETIONS)
        grid_polygons, ls_polygons = construct_grid(lattice, LANE_WIDTH, EDGE_WIDTH, self.off_params, LANE_SEP)
        self.road = []
        self.ls_polygons = ls_polygons

        # draw track
        i = 0
        for polygon in grid_polygons:
            t = self.world.CreateStaticBody(fixtures = fixtureDef(
                shape=polygonShape(vertices=list(polygon))
                ))
            t.userData = t
            c = 0.01*(i%3)
            i += 1
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.boxtype = "road"
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( list(polygon), t.color ))
            self.road.append(t)
        self.lattice = lattice
        self.track = grid_polygons

        # which vertices are in the lattice?
        h, w = GRID_ROWS, GRID_COLS
        self.which_points = []
        self.neighbors = []
        for i in range(h):
            for j in range(w):
                if lattice[i, j, 0]:
                    self.which_points += [(i, j)]
                    self.neighbors += [lattice[i, j, 1:]]

        # construct traffic lights
        self.lights = []
        for neighbor in self.neighbors:
            self.lights += [construct_traffic_lights(neighbor, LANE_WIDTH, TRAFFIC_LIGHT_R, TRAFFIC_LIGHT_R2)]

        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        
        # randomly init car
        ri, rj = self.which_points[np.random.randint(len(self.which_points))]
        y, x = ri*EDGE_WIDTH, rj*EDGE_WIDTH
        x, y = translate((x, y), self.off_params[1], self.off_params[0])
        self.car = Car(self.world, math.pi, x, y)

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count==len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=18,
                x=10, y=30,
                anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        # zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        # scroll_x = self.car.hull.position[0]
        # scroll_y = self.car.hull.position[1]
        # angle = -self.car.hull.angle
        # vel = self.car.hull.linearVelocity
        # if np.linalg.norm(vel) > 0.5:
        #    angle = math.atan2(vel[0], vel[1])
        # self.transform.set_scale(zoom, zoom)
        # self.transform.set_translation(
        #    WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
        #    WINDOW_H/2 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        # self.transform.set_rotation(angle)
        self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)

        # iterate through traffic lights
        for i, point in enumerate(self.which_points):
            shift_pos = self.off_params
            shift_pos = (shift_pos[1]+point[1]*EDGE_WIDTH, shift_pos[0]+point[0]*EDGE_WIDTH)
            for light in self.lights[i]:
                light_polygons = light.get_polygons(shift_pos) # should also step
                #self.viewer.draw_polygon(light_polygons[0], color=TRAFFIC_LIGHT_OUTER_COLOR)
                self.viewer.draw_polygon(light_polygons[1], color=TRAFFIC_LIGHT_INNER_COLOR)

        # lane separators
        for polygon in self.ls_polygons:
            self.viewer.draw_polygon(polygon, color=LANE_SEP_COLOR)

        self.car.draw(self.viewer, mode!="state_pixels")

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            image_data.save('tmp.png')

            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = CarGridDriving()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.close()

