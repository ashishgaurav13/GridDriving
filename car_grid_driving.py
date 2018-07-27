import sys, math
import numpy as np

import Box2D
from Box2D.b2 import edgeShape, circleShape
from Box2D.b2 import fixtureDef, polygonShape
from Box2D.b2 import revoluteJointDef
from friction_detector import FrictionDetector

import gym
from gym.envs.classic_control import rendering
from car_dynamics import Car
from gym.utils import colorize
from grid_utils import *
from env_utils import *

import pyglet
from pyglet import gl

# Initially created by Oleg Klimov.
# Licensed on the same terms as the rest of OpenAI Gym.
# Modified by Ashish Gaurav.

STATE_W, STATE_H = 256, 256
VIDEO_W, VIDEO_H = 900, 500
# Comment this out if you're not testing
# Also uncomment the transform part
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

NUM_VEHICLES = 2

ConvertRGB = lambda x: map(lambda y: y/255.0, x)
TRAFFIC_LIGHT_OUTER_COLOR = ConvertRGB((80, 126, 27))
TRAFFIC_LIGHT_INNER_COLOR = ConvertRGB((255, 255, 255))
ROAD_COLOR = [0.4, 0.4, 0.4]
LANE_SEP_COLOR = [0.6, 0.6, 0.6]

class CarGridDriving(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        
        # Init seed for randomness
        seed(self)
        
        # Contact Listener
        self.contactListener_keepref = FrictionDetector(self)
        
        # Create world
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        
        # Let's have a viewer for each vehicle since we need
        # a small image for each of the vehicle to train the
        # respective CNN/Network
        self.viewers = [None for i in range(NUM_VEHICLES)]
        self.cars = [None for i in range(NUM_VEHICLES)]

        # Holds all static bodies that make up the road, doesn't
        # include lane separators or traffic lights, just the
        # gray pieces
        self.road = None

        # TODO: decide how to use this reward
        self.rewards = make_n_rewards(NUM_VEHICLES)
        self.prev_rewards = make_n_rewards(NUM_VEHICLES)

        # action space has steer (-1, 0, 1), gas (0, 1), brake (0, 1)
        self.action_spaces = make_n_action_spaces(NUM_VEHICLES)

        # produce an observation space of rows x cols x 3
        state_dims = (STATE_H, STATE_W, 3)
        self.observation_spaces = make_n_state_spaces(NUM_VEHICLES, state_dims)

    def _destroy(self):

        # Destroy all the road bodies
        if self.road: 
            for t in self.road:
                self.world.DestroyBody(t)
            self.road = []

        # Destroy all cars
        for car in self.cars:
            if car:
                car.destroy()

    def _create_track(self):

        # Offset parameters to get the track correctly on the playfield
        self.off_params = (-WINDOW_H//4, -WINDOW_W//2.5)

        # Construct a lattice using a random node initialization followed
        # by a random edge deletion, ensuring everything is connected and
        # lattice constraints are met
        self.lattice = construct_lattice(GRID_ROWS, GRID_COLS, PROB_EDGE,
            LATTICE_CONSTRAINTS, RANDOM_DELETIONS)

        # Create polygons for the lattice road pieces and lane separators
        self.track, self.ls_polygons = construct_grid(self.lattice, LANE_WIDTH,
            EDGE_WIDTH, self.off_params, LANE_SEP)

        # Start with a blank list of road objects
        self.road = []

        # Draw track with varying shade so road pieces can be differentiated
        i = 0
        for polygon in self.track:

            # Create static body
            t = self.world.CreateStaticBody(fixtures = fixtureDef(
                shape=polygonShape(vertices=list(polygon))
                ))
            t.userData = t

            # Assign colors
            c = 0.01*(i%3)
            i += 1
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            
            # Assign properties and set contact true
            t.boxtype = "road"
            t.road_friction = 1.0
            t.fixtures[0].sensor = True

            # Add to list for rendering, and keep track by adding to self.road
            self.road_poly.append(( list(polygon), t.color ))
            self.road.append(t)

        # Which vertices are in the lattice?
        h, w = GRID_ROWS, GRID_COLS
        self.which_points = []
        self.neighbors = []
        for i in range(h):
            for j in range(w):
                if self.lattice[i, j, 0]:
                    self.which_points += [(i, j)]
                    self.neighbors += [self.lattice[i, j, 1:]]

        # Construct traffic lights
        self.lights = []
        for neighbor in self.neighbors:
            self.lights += [construct_traffic_lights(neighbor, LANE_WIDTH, TRAFFIC_LIGHT_R, TRAFFIC_LIGHT_R2)]

        return True

    def reset(self):

        # Destroy all objects
        self._destroy()

        # TODO: decide on the reward
        self.reward = 0.0
        self.prev_reward = 0.0
        
        # Transition for zoom
        self.t = make_n_times(NUM_VEHICLES)

        # Rendering values
        self.road_poly = []
        self.human_render = False

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        
        # randomly init each car
        car_idx = 0
        random_choices = np.random.choice(len(self.which_points), NUM_VEHICLES, replace=False)
        for rid in random_choices:
            ri, rj = self.which_points[rid]
            y, x = ri*EDGE_WIDTH, rj*EDGE_WIDTH
            x, y = translate((x, y), self.off_params[1], self.off_params[0])
            self.cars[car_idx] = Car(self.world, math.pi, x, y)
            car_idx += 1

        # return states after init
        return self.step([None for i in range(NUM_VEHICLES)])[0]

    # actions is a list of NUM_VEHICLES actions
    def step(self, actions):

        # apply the action to the available vehicles and step
        for car_idx, action in enumerate(actions):
            if action is not None:
                self.cars[car_idx].steer(-action[0])
                self.cars[car_idx].gas(action[1])
                self.cars[car_idx].brake(action[2])
            self.cars[car_idx].step(1.0/FPS)
            self.t[car_idx] += 1.0/FPS

        # Step the world
        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        # Step transition time for zoom

        # render images for each car, through their own viewer
        self.states = []
        for car_idx in range(NUM_VEHICLES):
            self.states.append(self.render(mode="state_pixels", car_idx=car_idx)) # TODO

        # find out reward for each of the cars
        step_rewards = make_n_rewards(NUM_VEHICLES)
        done_values = make_n_done_values(NUM_VEHICLES)
        for car_idx, action in enumerate(actions):
            if action is not None:

                # TODO : this and when is the env done?
                self.rewards[car_idx] -= 0.1
                
                # We actually don't want to count fuel spent, we want car to be faster.
                ## self.rewards[car_idx] -=  10 * self.cars[car_idx].fuel_spent / ENGINE_POWER
                self.cars[car_idx].fuel_spent = 0.0
                
                # Calculate this step's reward
                step_rewards[car_idx] = self.rewards[car_idx] - self.prev_rewards[car_idx]
                self.prev_rewards[car_idx] = self.rewards[car_idx]

                # If this car gets off playfield then huge negative reward and say its done
                x, y = self.cars[car_idx].hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done_values[car_idx] = True
                    step_rewards[car_idx] = -100

        return self.states, step_rewards, done_values, {}

    def render(self, car_idx, mode='human'):

        # Make the transforms and score labels if needed
        if "score_labels" not in self.__dict__:
            self.score_labels = [None for i in range(NUM_VEHICLES)]
        if "transforms" not in self.__dict__:
            self.transforms = [None for i in range(NUM_VEHICLES)]        

        # Construct a viewer for this car with score label and transform object
        if self.viewers[car_idx] is None:
            self.viewers[car_idx] = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_labels[car_idx] = pyglet.text.Label('0000', font_size=18,
            x=10, y=30,
            anchor_x='left', anchor_y='center',
            color=(255,255,255,255))
            self.transforms[car_idx] = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        # Create zoom effect and car following for this specific car
        zoom = 0.1*SCALE*max(1-self.t[car_idx], 0) + ZOOM*SCALE*min(self.t[car_idx], 1)   # Animate zoom first second
        scroll_x = self.cars[car_idx].hull.position[0]
        scroll_y = self.cars[car_idx].hull.position[1]
        angle = -self.cars[car_idx].hull.angle
        vel = self.cars[car_idx].hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
           angle = math.atan2(vel[0], vel[1])
        self.transforms[car_idx].set_scale(zoom, zoom)
        self.transforms[car_idx].set_translation(
           WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
           WINDOW_H/2 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transforms[car_idx].set_rotation(angle)

        # Comment out the block above and uncomment below if you want to see whole map
        # Also do the changes at the beginning of the file
        ## self.transforms[car_idx].set_translation(WINDOW_W/2, WINDOW_H/2)

        # Iterate through traffic lights
        # TODO: Should reduce these to bare minimum possible
        for i, point in enumerate(self.which_points):
            shift_pos = self.off_params
            shift_pos = (shift_pos[1]+point[1]*EDGE_WIDTH, shift_pos[0]+point[0]*EDGE_WIDTH)
            for light in self.lights[i]:

                light_polygons = light.get_polygons(shift_pos) # should also step

                # Uncomment if you want a circle outside traffic light
                ## self.viewer.draw_polygon(light_polygons[0], color=TRAFFIC_LIGHT_OUTER_COLOR)

                # Draw triangles for directions or square box for stop
                self.viewers[car_idx].draw_polygon(light_polygons[1], color=TRAFFIC_LIGHT_INNER_COLOR)

        # Lane separators
        for polygon in self.ls_polygons:
            self.viewers[car_idx].draw_polygon(polygon, color=LANE_SEP_COLOR)

        # Draw the car on the viewer
        self.cars[car_idx].draw(self.viewers[car_idx], mode!="state_pixels")

        # Do the actual rendering using pyglet.gl
        arr = None
        win = self.viewers[car_idx].window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transforms[car_idx]
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewers[car_idx].onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H, car_idx=car_idx)  # TODO: find why 2x needed
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            image_data.save('tmp%d.png'%car_idx)
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()
        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transforms[car_idx]
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewers[car_idx].onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H, car_idx=car_idx)
            win.flip()

        self.viewers[car_idx].onetime_geoms = []
        return arr

    # Close all viewers
    def close(self):
        for car_idx in range(NUM_VEHICLES):
            if self.viewers[car_idx] is not None:
                self.viewers[car_idx].close()
                self.viewers[car_idx] = None

    # Render all road pieces common to the whole environment
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

    # TODO: Some blocks in here are probably not needed
    def render_indicators(self, W, H, car_idx):
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
        true_speed = np.sqrt(np.square(self.cars[car_idx].hull.linearVelocity[0]) + np.square(self.cars[car_idx].hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.cars[car_idx].wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.cars[car_idx].wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.cars[car_idx].wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.cars[car_idx].wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.cars[car_idx].wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.cars[car_idx].hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_labels[car_idx].text = "%04i" % self.rewards[car_idx]
        self.score_labels[car_idx].draw()

if __name__=="__main__":
    from pyglet.window import key
    actions = [np.array( [0.0, 0.0, 0.0] ) for i in range(NUM_VEHICLES)]
    def key_press(k, mod, car_idx):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  actions[car_idx][0] = -1.0
        if k==key.RIGHT: actions[car_idx][0] = +1.0
        if k==key.UP:    actions[car_idx][1] = +1.0
        if k==key.DOWN:  actions[car_idx][2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod, car_idx):
        if k==key.LEFT  and actions[car_idx][0]==-1.0: actions[car_idx][0] = 0
        if k==key.RIGHT and actions[car_idx][0]==+1.0: actions[car_idx][0] = 0
        if k==key.UP:    actions[car_idx][1] = 0
        if k==key.DOWN:  actions[car_idx][2] = 0
    env = CarGridDriving()

    # Render a viewer for each car
    for car_idx in range(NUM_VEHICLES):
        env.render(car_idx=car_idx)
    
    # Let's hook key presses and releases to all viewers (TODO: this is a test)
    for car_idx in range(NUM_VEHICLES):
        env.viewers[car_idx].window.on_key_press = lambda k, mod: key_press(k, mod, car_idx)
        env.viewers[car_idx].window.on_key_release = lambda k, mod: key_release(k, mod, car_idx)
    
    # Loop on
    while True:
        env.reset()
        total_rewards = np.array(make_n_rewards(NUM_VEHICLES))
        steps = 0
        restart = False
        while True:
            states, rewards, done_values, info = env.step(actions)
            total_rewards += np.array(rewards)
            
            # TODO: figure out done values
            if steps % 200 == 0:
                print('alive')
                # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
        
            steps += 1
        
            # Render a viewer for each car
            for car_idx in range(NUM_VEHICLES):
                env.render(car_idx=car_idx)

            # if done or restart: break

    # End simulation
    env.close()

