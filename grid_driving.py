import sys, math
import numpy as np
from collections import deque

import Box2D
from Box2D import b2Vec2, b2Dot
from Box2D.b2 import edgeShape, circleShape
from Box2D.b2 import fixtureDef, polygonShape
from Box2D.b2 import revoluteJointDef

import gym
from gym.envs.classic_control import rendering
from gym.utils import colorize

import pyglet
from pyglet import gl
from pyglet.window import key

from constants import *
from env_utils import *
from friction_detector import FrictionDetector
from grid_utils import *
from localization import *
from car_dynamics import Car

class GridDriving(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    # For play mode
    def key_press(self, k, mod, car_idx):
        if k==key.LEFT:  self.extra_actions[car_idx][0] = -1.0
        if k==key.RIGHT: self.extra_actions[car_idx][0] = +1.0
        if k==key.UP:    self.extra_actions[car_idx][1] = +1.0
        if k==key.DOWN:  self.extra_actions[car_idx][2] = +0.8   # set 1.0 for wheels to block to zero rotation

    # For play mode
    def key_release(self, k, mod, car_idx):
        if k==key.LEFT  and self.extra_actions[car_idx][0]==-1.0: self.extra_actions[car_idx][0] = 0
        if k==key.RIGHT and self.extra_actions[car_idx][0]==+1.0: self.extra_actions[car_idx][0] = 0
        if k==key.UP:    self.extra_actions[car_idx][1] = 0
        if k==key.DOWN: self.extra_actions[car_idx][2] = 0

    def __init__(self, structure=None, structure_exclude=None, init_pos=None, rules=[], finish_pos=None, other_agents={}, play_mode=False, play_mode_idx=0):
        
        global NUM_VEHICLES
        NUM_VEHICLES = 1+len(other_agents)
        self.NUM_VEHICLES = NUM_VEHICLES
        self.rules = rules # TODO: move out
        assert(NUM_VEHICLES > 0)
        self.other_agents = other_agents
        self.pre_provided_lattice = structure
        self.delete_edges = structure_exclude
        self.play_mode = play_mode
        self.init_pos = init_pos
        self.finish_pos = finish_pos
        self.dist_eps = 20.0
        self.collision_eps = 10.0
        self.EDGE_WIDTH = EDGE_WIDTH
        self.DT = 1.0/FPS
        self.LANE_WIDTH = LANE_WIDTH
        self.last_positions = deque(maxlen=200) # TIMEOUT
        assert(play_mode_idx in range(NUM_VEHICLES))
        self.play_mode_car_idx = play_mode_idx

        if self.play_mode:
            self.extra_actions = [np.array( [0.0, 0.0, 0.0] ) for i in range(NUM_VEHICLES)]

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

        # Store tile direction, pid and relevant node for whatever tile was contacted last
        self.last_tile_direction = [None for i in range(NUM_VEHICLES)]
        self.last_pid = [None for i in range(NUM_VEHICLES)]
        self.last_rect_pid = [None for i in range(NUM_VEHICLES)]
        self.last_relevant_node = [None for i in range(NUM_VEHICLES)]

        # Hold localization strings
        self.loc = ["?" for i in range(NUM_VEHICLES)]
        self.tot_reward = [0.0 for _ in range(NUM_VEHICLES)]

        # Holds all static bodies that make up the road, doesn't
        # include lane separators or traffic lights, just the
        # gray pieces
        self.road = None

        # TODO: decide how to use this reward
        self.rewards = make_n_rewards(NUM_VEHICLES)
        self.prev_rewards = make_n_rewards(NUM_VEHICLES)
        self.infos = [None for i in range(NUM_VEHICLES)]

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
        if "lattice" not in self.__dict__:
            self.lattice = construct_lattice(GRID_ROWS, GRID_COLS, PROB_EDGE,
                LATTICE_CONSTRAINTS, RANDOM_DELETIONS, self.pre_provided_lattice, self.delete_edges)

        # Create polygons for the lattice road pieces and lane separators
        # Also store directions for each of the road piece
        if "track" not in self.__dict__:
            self.normal_polygons, self.special_polygons, self.ls_polygons, self.directions, \
            self.relevant_nodes, self.special_relevant_nodes = \
                construct_grid(self.lattice, LANE_WIDTH, EDGE_WIDTH, self.off_params, LANE_SEP)
            self.track = self.normal_polygons+self.special_polygons
            self.directions += "n"*len(self.special_polygons) # Each special_polygon is junction
            self.relevant_nodes += self.special_relevant_nodes
            assert(len(self.track) == len(self.directions) == len(self.relevant_nodes))

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

            # Assign direction
            t.direction = self.directions[i]

            # Assign colors
            # USEFUL: Change colors when debugging directions!
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
        h, w = self.lattice.shape[:2]
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
        self.tot_reward = [0.0 for _ in range(NUM_VEHICLES)]
        
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
        if not self.init_pos:
            rect_poly_indices = [i for i in range(len(self.directions)) if self.directions[i] in "lrtb"]
            random_choices = np.random.choice(rect_poly_indices, NUM_VEHICLES, replace=False)
            for car_idx, rid in enumerate(random_choices):
                rect_poly = np.array(self.road_poly[rid][0])
                direction = self.directions[rid]
                x = np.mean(rect_poly[:, 0])
                y = np.mean(rect_poly[:, 1])
                if direction == "r":
                    angle = -90
                elif direction == "t":
                    angle = 0
                elif direction == "l":
                    angle = 90
                else:
                    angle = 180
                self.cars[car_idx] = Car(self.world, angle*math.pi/180.0, x, y)
        else:
            for car_idx, init_p in enumerate(self.init_pos):
                i, j, angle = init_p
                i -= 0.5
                j += 0.5
                x, y = i*EDGE_WIDTH, j*EDGE_WIDTH
                x += self.off_params[0]
                y += self.off_params[1]
                self.cars[car_idx] = Car(self.world, angle*math.pi/180.0, x, y)

        # return states after init
        return self.step([None for i in range(NUM_VEHICLES)])[0]
        
    # actions is a list of NUM_VEHICLES actions
    # assign_rewards tells whether the car needs a sparse reward or not
    def step(self, actions):

        if self.play_mode:
            for car_idx in range(NUM_VEHICLES):
                if actions[car_idx] is not None and car_idx == self.play_mode_car_idx:
                    actions[car_idx] += self.extra_actions[car_idx]

        # apply the action to the available vehicles and step
        for car_idx, action in enumerate(actions):
            if car_idx != 0:
                if self.other_agents[str(car_idx)] == "stop":
                    action = np.array([0.0, 0.0, 1.0]) # BRAKE
            if action is not None:
                if action[0] <= -1: action[0] = -1
                if action[0] >= 1: action[0] = 1
                if action[1] <= 0: action[1] = 0
                if action[1] >= 1: action[1] = 1
                if action[2] <= 0: action[2] = 0
                if action[2] >= 1: action[2] = 1
                self.cars[car_idx].steer(-action[0])
                self.cars[car_idx].gas(action[1])
                self.cars[car_idx].brake(action[2])
            self.cars[car_idx].step(1.0/FPS)
            self.t[car_idx] += 1.0/FPS

        # Step the world
        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        # Regenerate localization info
        min_speed = 2
        for car_idx in range(NUM_VEHICLES):
            on_road, pid = self.determine_onroad(car_idx)
            self.last_pid[car_idx] = pid
            if on_road:
                self.last_tile_direction[car_idx] = self.directions[pid]
                # Also update last relevant node
                self.last_relevant_node[car_idx] = self.relevant_nodes[pid]
            else:
                self.loc[car_idx] = "off-road"
                continue
            if self.directions[pid] in ["l", "r", "t", "b"]:
                self.last_rect_pid[car_idx] = pid
            if self.last_tile_direction[car_idx] == None:
                continue
            vx, vy = self.cars[car_idx].hull.linearVelocity
            theta = self.cars[car_idx].hull.angle % np.pi
            # What is the right direction
            right_dir = self.last_tile_direction[car_idx]
            if right_dir == "l":
                if vx < -min_speed or np.pi/4 <= theta < 3*np.pi/4:
                    self.loc[car_idx] = "right"
                elif vx > min_speed or 5*np.pi/4 <= theta < 7*np.pi/4:
                    self.loc[car_idx] = "left"
            elif right_dir == "r":
                if vx > min_speed or 5*np.pi/4 <= theta < 7*np.pi/4:
                    self.loc[car_idx] = "right"
                elif vx < -min_speed or np.pi/4 <= theta < 3*np.pi/4:
                    self.loc[car_idx] = "left"
            elif right_dir == "t":
                if vy > min_speed or (0 <= theta < np.pi/4 or 7*np.pi/4 <= theta < 2*np.pi):
                    self.loc[car_idx] = "right"
                elif vy < -min_speed or 3*np.pi/4 <= theta < 5*np.pi/4:
                    self.loc[car_idx] = "left"
            elif right_dir == "b":
                if vy < -min_speed or 3*np.pi/4 <= theta < 5*np.pi/4:
                    self.loc[car_idx] = "right"
                elif vy > min_speed or (0 <= theta < np.pi/4 or 7*np.pi/4 <= theta < 2*np.pi):
                    self.loc[car_idx] = "left"
            elif right_dir == "n":
                # TODO: Assuming there are no 1-edge junctions
                self.loc[car_idx] = "junction"
        
        # render images for each car, through their own viewer
        self.states = []
        empty_dict = {
            'on_rect': None,
            'lane_localization': None,
            'traffic_light': None,
            'off_road': None,
            'type_intersection': None,
            'junction': None,
            'speed': None,
            'pos': None,
        }
        for car_idx in range(NUM_VEHICLES):
            rstate, values = self.render(mode="state_pixels", car_idx=car_idx)
            self.states.append(rstate) # TODO
            pos = self.cars[car_idx].hull.position
            vel = self.cars[car_idx].hull.linearVelocity
            speed = np.sqrt(vel[0]**2+vel[1]**2)
            self.infos[car_idx] = dict(empty_dict)
            # self.infos[car_idx]['lane_localization'] = self.loc[car_idx]
            # self.infos[car_idx]['on_rect'] = self.loc[car_idx] in ['left', 'right']
            if self.loc[car_idx] in ['left', 'right']:
                self.infos[car_idx]['lane'] = self.loc[car_idx]
            self.infos[car_idx]['off_road'] = self.loc[car_idx] == 'off-road'
            # self.infos[car_idx]['junction'] = self.loc[car_idx] == 'junction'
            if values != None:
                self.infos[car_idx]['traffic_light'] = values[0]
                self.infos[car_idx]['type_intersection'] = values[1]
                self.infos[car_idx]['only_turn'] = values[2] if self.loc[car_idx] == 'right' else None
                # self.infos[car_idx]['junction_pos'] = values[3]
            self.infos[car_idx]['v'] = speed
            self.infos[car_idx]['vx'] = vel.x
            self.infos[car_idx]['vy'] = vel.y
            # self.infos[car_idx]['pos'] = (pos.x, pos.y)
            self.infos[car_idx]['x'] = pos.x
            self.infos[car_idx]['y'] = pos.y
            # self.infos[car_idx]['last_rect'] = tuple(map(tuple, self.road_poly[self.last_pid[car_idx]][0]))

        # Reward Assignment (sparse)
        step_rewards = [0 for i in range(len(actions))]
        done_values = [False for i in range(len(actions))]
        for car_idx, action in enumerate(actions):
               
                step_rewards[car_idx] = 0
                
                # We actually don't want to count fuel spent, we want car to be faster.
                ## self.rewards[car_idx] -=  10 * self.cars[car_idx].fuel_spent / ENGINE_POWER
                self.cars[car_idx].fuel_spent = 0.0
                
                # If this car gets off playfield
                x, y = self.cars[car_idx].hull.position

                # Or if close to finish coordinates
                if self.finish_pos and car_idx == 0:
                    i, j = self.finish_pos
                    i -= 0.5
                    j += 0.5
                    fx, fy = i*EDGE_WIDTH, j*EDGE_WIDTH
                    fx += self.off_params[0]
                    fy += self.off_params[1]
                    dist = ((fx-x)**2+(fy-y)**2)**0.5
                    if dist < self.dist_eps:
                        done_values[car_idx] = True

                # Check collision
                if car_idx == 0 and NUM_VEHICLES > 1:
                    for ci in range(1, NUM_VEHICLES):
                        cx, cy = self.cars[ci].hull.position
                        x, y = self.cars[car_idx].hull.position
                        dist = ((x-cx)**2+(y-cy)**2)**0.5
                        if dist < self.collision_eps:
                            print('COLLISION: dist to another veh = %f' % dist)
                            done_values[car_idx] = True

                # If position didn't change then terminate
                if car_idx == 0:
                    x, y = self.cars[car_idx].hull.position
                    x -= self.off_params[0]
                    y -= self.off_params[1]
                    x /= self.EDGE_WIDTH
                    y /= self.EDGE_WIDTH
                    self.last_positions.append((x, y))
                if car_idx == 0 and len(self.last_positions) == self.last_positions.maxlen:
                    x1, y1 = self.last_positions[0]
                    x2, y2 = self.last_positions[-1]
                    d = ((x1-x2)**2+(y1-y2)**2)**0.5
                    if d < 0.02:
                        print('TIMEOUT: (%f,%f)->(%f,%f)' % (x1, y1, x2, y2))
                        done_values[car_idx] = True

                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done_values[car_idx] = True

        for car_idx in range(NUM_VEHICLES):
            self.tot_reward[car_idx] += step_rewards[car_idx]

        # Update score labels
        for car_idx in range(NUM_VEHICLES):
            if self.viewers[car_idx] != None:
                self.score_labels[car_idx][0].text = 'traffic_light: %s' % self.infos[car_idx]['traffic_light']
                self.score_labels[car_idx][1].text = 'lane_localization: %s' % self.infos[car_idx]['lane_localization']
                self.score_labels[car_idx][2].text = 'type_intersection: %s' % self.infos[car_idx]['type_intersection']
                self.score_labels[car_idx][3].text = 'only_turn: %s' % self.infos[car_idx]['only_turn']

        # Remove None(s), more infos formatting
        for car_idx in range(NUM_VEHICLES):
            final_dict = {}
            for key in self.infos[car_idx].keys():
                if self.infos[car_idx][key] != None:
                    final_dict[key] = self.infos[car_idx][key]
                    if type(final_dict[key]) in [float, np.float64]:
                        final_dict[key] = round(final_dict[key], 2)
            self.infos[car_idx] = final_dict
        self.infos = {car_idx:self.infos[car_idx] for car_idx in range(NUM_VEHICLES)}

        return self.states, step_rewards, done_values, self.infos

    # Determine if the car is offroad or onroad
    def determine_onroad(self, car_idx):
        return determine_road(self.lattice, EDGE_WIDTH, self.road_poly,
            self.cars[car_idx].hull.position)

    def render(self, car_idx=None, mode='human', pts=None):

        # If car_idx = None, then all cars should be shown in different windows
        if car_idx is None:
            self.render(0, mode, pts)
            # Uncomment for multiple viewers
            # for i in range(NUM_VEHICLES):
            #     self.render(i, mode, pts)
            return

        # Make the transforms and score labels if needed
        if "score_labels" not in self.__dict__:
            self.score_labels = [[] for i in range(NUM_VEHICLES)]
        if "transforms" not in self.__dict__:
            self.transforms = [None for i in range(NUM_VEHICLES)]        

        # Construct a viewer for this car with score label and transform object
        if self.viewers[car_idx] is None and car_idx == 0:
            self.viewers[car_idx] = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_labels[car_idx].append(pyglet.text.Label('traffic_light: ?', font_size=12,
            x=10, y=80,
            anchor_x='left', anchor_y='center', font_name='Helvetica',
            color=(255,255,255,255)))
            self.score_labels[car_idx].append(pyglet.text.Label('lane_localization: ?', font_size=12,
            x=10, y=60,
            anchor_x='left', anchor_y='center', font_name='Helvetica',
            color=(255,255,255,255)))
            self.score_labels[car_idx].append(pyglet.text.Label('type_intersection: ?', font_size=12,
            x=10, y=40,
            anchor_x='left', anchor_y='center', font_name='Helvetica',
            color=(255,255,255,255)))
            self.score_labels[car_idx].append(pyglet.text.Label('only_turn: ?', font_size=12,
            x=10, y=20,
            anchor_x='left', anchor_y='center', font_name='Helvetica',
            color=(255,255,255,255)))
            self.transforms[car_idx] = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet
        if car_idx != 0: return None, None

        # Create zoom effect and car following for this specific car
        # zoom = 0.1*SCALE*max(1-self.t[car_idx], 0) + ZOOM*SCALE*min(self.t[car_idx], 1)   # Animate zoom first second
        zoom = ZOOM*SCALE
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
        # self.transforms[car_idx].set_translation(WINDOW_W/2, WINDOW_H/2)

        # Iterate through traffic lights
        # We only want to show the relevant traffic lights
        # 1. Find out which node in the lattice is relevant
        # 2. Find out traffic lights in current rectangle (if we have r/l lane classification)
        # 3. Show just that
        showed_traffic_light = None
        count_lights = None
        shift_pos = None
        if self.last_pid[car_idx] != -1:
            relevant_node = self.last_relevant_node[car_idx]
            classification_val = self.loc[car_idx] == "right"
            shift_pos = self.off_params
            shift_pos = (
                shift_pos[1]+relevant_node[1]*EDGE_WIDTH,
                shift_pos[0]+relevant_node[0]*EDGE_WIDTH
            )
            if classification_val:
                i = -1
                if relevant_node in self.which_points:
                    i = self.which_points.index(relevant_node)
                if i != -1:
                    for light in self.lights[i]:

                        light_polygons = light.get_polygons(shift_pos) # should also step

                        traffic_light_pos = Point(light.shifted_pos(shift_pos))
                        our_polygon = Polygon(self.road_poly[self.last_pid[car_idx]][0])
                        # print(light.pos, self.road_poly[self.last_pid[car_idx]][0])
                        if not our_polygon.contains(traffic_light_pos):
                            continue
                        # Uncomment if you want a circle outside traffic light
                        ## self.viewer.draw_polygon(light_polygons[0], color=TRAFFIC_LIGHT_OUTER_COLOR)

                        # Draw triangles for directions or square box for stop
                        showed_traffic_light = light.state_cycle[light.state_idx]
                        count_lights = len(self.lights[i])
                        self.viewers[car_idx].draw_polygon(light_polygons[1], color=TRAFFIC_LIGHT_INNER_COLOR)

        # Get the only possible turn at this junction
        my_last_rect_pid = self.last_rect_pid[car_idx]
        only_turn = None
        if my_last_rect_pid and my_last_rect_pid != -1:
            rni, rnj = self.last_relevant_node[car_idx]
            neighbor_info = list(self.lattice[rni, rnj, 1:])
            my_dir_last = self.directions[my_last_rect_pid]
            if np.sum(neighbor_info) == 2:
                if neighbor_info == [False, False, True, True]:
                    if my_dir_last == 'r':
                        only_turn = "right"
                    elif my_dir_last == 't':
                        only_turn = "left"
                elif neighbor_info == [True, True, False, False]:
                    if my_dir_last == 'l':
                        only_turn = "right"
                    elif my_dir_last == 'b':
                        only_turn = "left"
                elif neighbor_info == [False, True, True, False]:
                    if my_dir_last == 'l':
                        only_turn = "left"
                    elif my_dir_last == 't':
                        only_turn = "right"
                elif neighbor_info == [True, False, False, True]:
                    if my_dir_last == 'r':
                        only_turn = "left"
                    elif my_dir_last == 'b':
                        only_turn = "right"

        # Lane separators
        for polygon in self.ls_polygons:
            self.viewers[car_idx].draw_polygon(polygon, color=LANE_SEP_COLOR)

        # Draw all cars on the viewer
        for ci in range(NUM_VEHICLES):
            self.cars[ci].draw(self.viewers[car_idx], mode!="state_pixels")

        # Do the actual rendering using pyglet.gl
        arr = None
        win = self.viewers[car_idx].window

        # If play mode, then assign key press and key release events
        if self.play_mode:
            if car_idx == self.play_mode_car_idx:
                win.on_key_press = lambda k, mod: self.key_press(k, mod, car_idx)
                win.on_key_release = lambda k, mod: self.key_release(k, mod, car_idx)


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
            # self.show_risk()
            self.render_additional_points(pts)
            for geom in self.viewers[car_idx].onetime_geoms:
                geom.render()
            t.disable()
            # self.render_indicators(WINDOW_W, WINDOW_H, car_idx=car_idx)  # TODO: find why 2x needed
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            # image_data.save('tmp%d.png'%car_idx)
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3].astype(np.float64)
            # arr = rgb2gray(arr)
            # arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
            arr /= 255.0
        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()
        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transforms[car_idx]
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            # self.show_risk()
            self.render_additional_points(pts)
            for geom in self.viewers[car_idx].onetime_geoms:
                geom.render()
            t.disable()
            # self.render_indicators(WINDOW_W, WINDOW_H, car_idx=car_idx)
            win.flip()

        self.viewers[car_idx].onetime_geoms = []
        return arr, [showed_traffic_light, count_lights if count_lights else None, only_turn, shift_pos]

    # Close all viewers
    def close(self):
        for car_idx in range(NUM_VEHICLES):
            if self.viewers[car_idx] is not None:
                self.viewers[car_idx].close()
                self.viewers[car_idx] = None

    # Get all important attributes (freeze/microwave)
    def freeze(self):
        exc = ['world', 'cars', 'road', 'score_labels', 'contactListener_keepref', 'viewers']
        return deepcopy_env(self, exc)

    def microwave(self, x):
        self.close()
        failures = []
        for frozen_attr in dir(x):
            try:
                setattr(self, frozen_attr, getattr(x, frozen_attr))
            except:
                failures.append(frozen_attr)
        # print('Microwave failures: %s' % failures)

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.cars = [None for car_idx in range(self.NUM_VEHICLES)]
        self.viewers = [None for car_idx in range(self.NUM_VEHICLES)]
        self.score_labels = [[] for i in range(self.NUM_VEHICLES)]
        self.transforms = [None for i in range(self.NUM_VEHICLES)]

        self.road = []
        i = 0
        for polygon in self.track:
            t = self.world.CreateStaticBody(fixtures = fixtureDef(
                shape=polygonShape(vertices=list(polygon))
                ))
            t.userData = t
            t.direction = self.directions[i]
            c = 0.01*(i%3)
            i += 1
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.boxtype = "road"
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road.append(t)

        # print(self.car_info)
        for car_idx in range(self.NUM_VEHICLES):
            x, y = self.car_info[car_idx]['pos']
            angle = self.car_info[car_idx]['angle']
            self.cars[car_idx] = Car(self.world, angle, x, y)
            self.cars[car_idx].linearVelocity = self.car_info[car_idx]['v']
            self.cars[car_idx].angularVelocity = self.car_info[car_idx]['av']
            for wi, w in enumerate(self.car_info[car_idx]['w']):
                gas, brake, steer = w
                self.cars[car_idx].wheels[wi].gas = gas
                self.cars[car_idx].wheels[wi].brake = brake
                self.cars[car_idx].wheels[wi].steer = steer

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

    def render_additional_points(self, pts):
        if pts == None: return
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        for (x, y) in pts:
            gl.glColor4f(0, 0, 1.0, 1.0)
            gl.glVertex2f(x, y, 0)
        gl.glEnd()

    def show_risk(self):
        # print('%s' % self.infos)
        # print("%s" % hasattr(self, 'reward_strategy'))
        if self.infos[0] != None and self.infos[0]['on_rect'] and hasattr(self, 'reward_strategy'):
            # win = self.env.viewers[0].window
            # win.switch_to()
            # win.dispatch_events()
            rect = self.infos[0]['last_rect']
            rectx = sorted(list(set(map(lambda item: item[0], rect)))) # unique x
            recty = sorted(list(set(map(lambda item: item[1], rect)))) # unique y
            # gl.glViewport(0, 0, 96, 96)
            gl.glPointSize(3.3)
            gl.glBegin(gl.GL_POINTS)
            oc = [0.5, 0.5, 0.5]
            gl.glColor4f(*oc, 1.0)
            x = rectx[0]
            while x < rectx[1]:
                y = recty[0]
                while y < recty[1]:
                    r = self.reward_strategy.risk(x, y)
                    gl.glColor4f(r*oc[0], r*oc[1], r*oc[2], 1.0)
                    print("%s = %s" % ((x, y), r))
                    gl.glVertex2f(x, y, 0)
                    y += 1.0
                x += 1.0
            gl.glEnd()

    # TODO: Some blocks in here are probably not needed
    def render_indicators(self, W, H, car_idx):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        # gl.glVertex3f(W, 5*h, 0)
        # gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(W, H/5.0, 0)
        gl.glVertex3f(0, H/5.0, 0)
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
        # vertical_ind(5, 0.02*true_speed, (1,1,1))
        # vertical_ind(7, 0.01*self.cars[car_idx].wheels[0].omega, (0.0,0,1)) # ABS sensors
        # vertical_ind(8, 0.01*self.cars[car_idx].wheels[1].omega, (0.0,0,1))
        # vertical_ind(9, 0.01*self.cars[car_idx].wheels[2].omega, (0.2,0,1))
        # vertical_ind(10,0.01*self.cars[car_idx].wheels[3].omega, (0.2,0,1))
        # horiz_ind(20, -10.0*self.cars[car_idx].wheels[0].joint.angle, (0,1,0))
        # horiz_ind(30, -0.8*self.cars[car_idx].hull.angularVelocity, (1,0,0))
        gl.glEnd()
        for el in self.score_labels[car_idx]:
            el.draw()