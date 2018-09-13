import sys, math
import numpy as np

import Box2D
from Box2D import b2Vec2, b2Dot
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
from localization import *

import pyglet
from pyglet import gl
from constants import *

class CarGridDriving(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, options):
        
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

        # action space has steer (-1, 0, 1), gas (0, 1), brake (0, 1)
        self.action_spaces = make_n_action_spaces(NUM_VEHICLES)

        # produce an observation space of rows x cols x 3
        state_dims = (STATE_H, STATE_W, 3)
        self.observation_spaces = make_n_state_spaces(NUM_VEHICLES, state_dims)

        # options
        self.options = options
        self.curr_node = None

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
                LATTICE_CONSTRAINTS, RANDOM_DELETIONS)

        # Create polygons for the lattice road pieces and lane separators
        # Also store directions for each of the road piece
        if "track" not in self.__dict__:
            self.track, self.ls_polygons, self.directions, self.relevant_nodes = \
                construct_grid(self.lattice, LANE_WIDTH, EDGE_WIDTH, self.off_params, LANE_SEP)

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
            
        # return states after init
        return self.step([None for i in range(NUM_VEHICLES)])[0]
        
    # actions is a list of NUM_VEHICLES actions
    # curr_node stores node info in options graph
    # assign_rewards tells whether the car needs a sparse reward or not
    def step(self, actions, curr_node=None):

        # apply the action to the available vehicles and step
        for car_idx, action in enumerate(actions):
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
            if self.directions[pid] in ["l", "r", "t", "b"]:
                self.last_rect_pid[car_idx] = pid
            if on_road:
                self.last_tile_direction[car_idx] = self.directions[pid]
                # Also update last relevant node
                self.last_relevant_node[car_idx] = self.relevant_nodes[pid]
            else:
                self.loc[car_idx] = "off-road"
                continue
            if self.last_tile_direction[car_idx] == None:
                continue
            vx, vy = self.cars[car_idx].hull.linearVelocity
            # What is the right direction
            right_dir = self.last_tile_direction[car_idx]
            if right_dir == "l":
                if vx < -min_speed:
                    self.loc[car_idx] = "right"
                elif vx > min_speed:
                    self.loc[car_idx] = "left"
            elif right_dir == "r":
                if vx > min_speed:
                    self.loc[car_idx] = "right"
                elif vx < -min_speed:
                    self.loc[car_idx] = "left"
            elif right_dir == "t":
                if vy > min_speed:
                    self.loc[car_idx] = "right"
                elif vy < -min_speed:
                    self.loc[car_idx] = "left"
            elif right_dir == "b":
                if vy < -min_speed:
                    self.loc[car_idx] = "right"
                elif vy > min_speed:
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
        self.infos = [dict(empty_dict) for i in range(NUM_VEHICLES)]
        for car_idx in range(NUM_VEHICLES):
            rstate, values = self.render(mode="state_pixels", car_idx=car_idx)
            self.states.append(rstate) # TODO
            pos = self.cars[car_idx].hull.position
            vel = self.cars[car_idx].hull.linearVelocity
            speed = np.sqrt(vel[0]**2+vel[1]**2)
            self.infos[car_idx]['traffic_light'] = values[0]
            self.infos[car_idx]['lane_localization'] = self.loc[car_idx]
            self.infos[car_idx]['on_rect'] = self.loc[car_idx] in ['left', 'right']
            self.infos[car_idx]['off_road'] = self.loc[car_idx] == 'off-road'
            self.infos[car_idx]['junction'] = self.loc[car_idx] == 'junction'
            self.infos[car_idx]['type_intersection'] = values[1]
            self.infos[car_idx]['only_turn'] = values[2]
            self.infos[car_idx]['speed'] = speed
            self.infos[car_idx]['pos'] = pos
            self.infos[car_idx]['last_rect'] = tuple(map(tuple, self.road_poly[self.last_pid[car_idx]][0]))

        # find out reward for each of the cars
        drive_straight = 1
        decelerate_to_halt = 2
        stop_and_wait3 = [3, 5]
        stop_and_wait4 = [3, 4, 5]
        left_on_junc = 6
        straight_on_junc = 7
        right_on_junc = 8
        off_road = 9
        step_rewards = make_n_rewards(NUM_VEHICLES)
        done_values = make_n_done_values(NUM_VEHICLES)

        # Reward Assignment (sparse)
        for car_idx, action in enumerate(actions):
            if action is not None:
                
                step_rewards[car_idx] = 0

                # N6, N7, N8: entering (determine target_pos)
                if self.options.entering_node6[car_idx] == True or \
                   self.options.entering_node7[car_idx] == True or \
                   self.options.entering_node8[car_idx] == True:

                    if self.cars[car_idx] and self.cars[car_idx].hull:
                        x, y = self.cars[car_idx].hull.position
                        relevant_node = self.last_relevant_node[car_idx]
                        i = -1
                        if relevant_node in self.which_points:
                            i = self.which_points.index(relevant_node)
                        if i != -1:
                            x0, y0 = relevant_node[1]*EDGE_WIDTH, relevant_node[0]*EDGE_WIDTH
                            lx, ly = x0-2*LANE_WIDTH, y0+0.5*LANE_WIDTH
                            rx, ry = x0+2*LANE_WIDTH, y0-0.5*LANE_WIDTH
                            sx, sy = x0+0.5*LANE_WIDTH, y0+2*LANE_WIDTH
                            px, py = x0-0.5*LANE_WIDTH, y0-2*LANE_WIDTH
                            dl = dist(x, y, lx, ly)
                            dr = dist(x, y, rx, ry)
                            ds = dist(x, y, sx, sy)
                            dp = dist(x, y, px, py)
                            # entering from left side of intersection
                            if dl == min(dl, dr, ds, dp):
                                if self.options.entering_node6[car_idx] == True:
                                    self.options.target_pos_node6[car_idx] = (sx, sy)
                                if self.options.entering_node7[car_idx] == True:
                                    self.options.target_pos_node7[car_idx] = (rx, ry)
                                if self.options.entering_node8[car_idx] == True:
                                    self.options.target_pos_node8[car_idx] = (px, py)
                            # entering from top side of intersection
                            if ds == min(dl, dr, ds, dp):
                                if self.options.entering_node6[car_idx] == True:
                                    self.options.target_pos_node6[car_idx] = (rx, ry)
                                if self.options.entering_node7[car_idx] == True:
                                    self.options.target_pos_node7[car_idx] = (px, py)
                                if self.options.entering_node8[car_idx] == True:
                                    self.options.target_pos_node8[car_idx] = (lx, ly)
                            # entering from right side of intersection
                            if dr == min(dl, dr, ds, dp):
                                if self.options.entering_node6[car_idx] == True:
                                    self.options.target_pos_node6[car_idx] = (px, py)
                                if self.options.entering_node7[car_idx] == True:
                                    self.options.target_pos_node7[car_idx] = (lx, ly)
                                if self.options.entering_node8[car_idx] == True:
                                    self.options.target_pos_node8[car_idx] = (sx, sy)
                            # entering from bottom side of intersection
                            if dp == min(dl, dr, ds, dp):
                                if self.options.entering_node6[car_idx] == True:
                                    self.options.target_pos_node6[car_idx] = (lx, ly)
                                if self.options.entering_node7[car_idx] == True:
                                    self.options.target_pos_node7[car_idx] = (sx, sy)
                                if self.options.entering_node8[car_idx] == True:
                                    self.options.target_pos_node8[car_idx] = (rx, ry)



                # not initialized properly
                if curr_node is None:
                    continue
                else:
                    self.curr_node = curr_node

                # N1: driving straight
                # check node1 assign rewards
                #
                # What happens if the car doesn't move?
                if curr_node[car_idx] == drive_straight or self.options.assign_node1[car_idx] == True:
                    # minx, miny = np.inf, np.inf
                    # maxx, maxy = -np.inf, -np.inf
                    # for pos in self.options.pos_node1[car_idx]:
                    #     minx, miny = min(minx, pos[0]), min(miny, pos[1])
                    #     maxx, maxy = max(maxx, pos[0]), max(maxy, pos[1])
                    # movement_dir = 0 if maxx-minx > maxy-miny else 1
                    # xr, yr = 0, 0
                    # rect_list = list(self.options.last_rect_node1[car_idx])
                    # xset, yset = set(), set()
                    # for rindex in range(len(rect_list)):
                    #     if not isLaneRect(rect_list[rindex]):
                    #         continue
                    #     for pos in rect_list[rindex]:
                    #         xset.add(pos[0])
                    #         yset.add(pos[1])
                    # if movement_dir == 0:
                    #     target_distance = 0.01 # instead of 0 to avoid handling that bad case of 0/0 division
                    #     if len(xset) > 0:
                    #         target_distance = max(xset)-min(xset)
                    #     covered_distance = 0.0
                    #     if len(self.options.pos_node1[car_idx]) > 0:
                    #         covered_distance = maxx-minx
                    #     step_rewards[car_idx] = ((covered_distance)*5.0)/(target_distance)
                    # else:
                    #     target_distance = 0.01 # instead of 0 to avoid handling that bad case of 0/0 division
                    #     if len(yset) > 0:
                    #         target_distance = max(yset)-min(yset)
                    #     covered_distance = 0.0
                    #     if len(self.options.pos_node1[car_idx]) > 0:
                    #         covered_distance = maxy-miny
                    #     step_rewards[car_idx] = ((covered_distance)*5.0)/(target_distance)

                    if len(self.options.pos_node1[car_idx]) > 1:    
                        # find out displacement in x direction, y direction
                        last_pt = self.options.pos_node1[car_idx][-1]
                        second_last_pt = self.options.pos_node1[car_idx][-2]
                        xdisp, ydisp = last_pt[0]-second_last_pt[0], last_pt[1]-second_last_pt[1]
                        displacement = b2Vec2(xdisp, ydisp)
                        # localize last point
                        rect_list = list(self.options.last_rect_node1[car_idx])
                        correct_index = -1
                        for rindex in range(len(rect_list)):
                            if Polygon(rect_list[rindex]).contains(Point(last_pt[0], last_pt[1])):
                                correct_index = rindex
                                break
                        if correct_index != -1:
                            xset, yset = set(), set()
                            for pt in rect_list[rindex]:
                                xset.add(pt[0])
                                yset.add(pt[1])
                            xr, yr = max(xset)-min(xset), max(yset)-min(yset)
                            movement_dir = 0 if yr == LANE_WIDTH else 1
                            if self.infos[car_idx]['speed'] >= 10:
                                if movement_dir == 0:
                                    if self.options.direction_node1[car_idx] is None:
                                        possible_dir = b2Vec2((1, 0))
                                        veh_pointing = self.cars[car_idx].get_acc_direction()
                                        dot_prod = b2Dot(possible_dir, veh_pointing)
                                        if dot_prod > 0: # (1, 0) is the right direction
                                            self.options.direction_node1[car_idx] = possible_dir
                                        else:
                                            self.options.direction_node1[car_idx] = b2Vec2((-1, 0))
                                    else:
                                        correct_direction = self.options.direction_node1[car_idx]
                                        dot_value = b2Dot(displacement, correct_direction)
                                        step_rewards[car_idx] = 1.0*np.abs(dot_value)/(xr)
                                        multiplier = 1 if dot_value > 0 and self.infos[car_idx]['lane_localization'] == "right" else -5
                                        step_rewards[car_idx] *= multiplier
                                else:
                                    if self.options.direction_node1[car_idx] is None:
                                        possible_dir = b2Vec2((0, 1))
                                        veh_pointing = self.cars[car_idx].get_acc_direction()
                                        dot_prod = b2Dot(possible_dir, veh_pointing)
                                        if dot_prod > 0: # (0, 1) is the right direction
                                            self.options.direction_node1[car_idx] = possible_dir
                                        else:
                                            self.options.direction_node1[car_idx] = b2Vec2((0, -1))
                                    else:
                                        correct_direction = self.options.direction_node1[car_idx]
                                        dot_value = b2Dot(displacement, correct_direction)
                                        step_rewards[car_idx] = 1.0*np.abs(dot_value)/(yr)
                                        multiplier = 1 if dot_value > 0 and self.infos[car_idx]['lane_localization'] == "right" else -5
                                        step_rewards[car_idx] *= multiplier

                    # if entering node9, penalize
                    if self.options.entering_node9[car_idx] == True:
                        step_rewards[car_idx] -= 3

                # N2: decel to stop
                elif self.options.assign_node2[car_idx] == True:
                    v2s = self.options.start_vel_node2[car_idx]
                    v2e = self.options.end_vel_node2[car_idx]
                    step_rewards[car_idx] = (v2s-v2e)

                # N3: stop
                elif self.options.assign_node3[car_idx] == True:
                    step_rewards[car_idx] = self.options.num_brakes_node3[car_idx]

                # N4: stop
                elif self.options.assign_node4[car_idx] == True:
                    step_rewards[car_idx] = self.options.num_brakes_node4[car_idx]

                # N5: stop
                elif self.options.assign_node5[car_idx] == True:
                    step_rewards[car_idx] = self.options.num_brakes_node5[car_idx]

                # N6: turn left
                elif self.options.assign_node6[car_idx] == True:
                    if self.cars[car_idx] and self.cars[car_idx].hull:
                        x, y = self.cars[car_idx].hull.position
                        eps = LANE_WIDTH//2
                        tx, ty = self.options.target_pos_node6[car_idx]
                        if dist(x, y, tx, ty) <= eps:
                            step_rewards[car_idx] = 3
                        else:
                            step_rewards[car_idx] = -3

                # N7: move straight
                elif self.options.assign_node7[car_idx] == True:
                    if self.cars[car_idx] and self.cars[car_idx].hull:
                        x, y = self.cars[car_idx].hull.position
                        eps = LANE_WIDTH//2
                        tx, ty = self.options.target_pos_node7[car_idx]
                        if dist(x, y, tx, ty) <= eps:
                            step_rewards[car_idx] = 3
                        else:
                            step_rewards[car_idx] = -3

                # N8: turn right
                elif self.options.assign_node8[car_idx] == True:
                    if self.cars[car_idx] and self.cars[car_idx].hull:
                        x, y = self.cars[car_idx].hull.position
                        eps = LANE_WIDTH//2
                        tx, ty = self.options.target_pos_node8[car_idx]
                        if dist(x, y, tx, ty) <= eps:
                            step_rewards[car_idx] = 3
                        else:
                            step_rewards[car_idx] = -3

                elif curr_node[car_idx] == off_road:
                    step_rewards[car_idx] = -10

                
                # We actually don't want to count fuel spent, we want car to be faster.
                ## self.rewards[car_idx] -=  10 * self.cars[car_idx].fuel_spent / ENGINE_POWER
                self.cars[car_idx].fuel_spent = 0.0
                
                # Calculate this step's reward
                # step_rewards[car_idx] = self.rewards[car_idx] - self.prev_rewards[car_idx]
                # self.prev_rewards[car_idx] = self.rewards[car_idx]

                # If this car gets off playfield then huge negative reward and say its done
                x, y = self.cars[car_idx].hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done_values[car_idx] = True
                    step_rewards[car_idx] = -100

        for car_idx in range(NUM_VEHICLES):
            self.tot_reward[car_idx] += step_rewards[car_idx]
        return self.states, step_rewards, done_values, self.infos

    # Determine if the car is offroad or onroad
    def determine_onroad(self, car_idx):
        return determine_road(self.lattice, EDGE_WIDTH, self.road_poly,
            self.cars[car_idx].hull.position)

    def render(self, car_idx, mode='human'):

        # Make the transforms and score labels if needed
        if "score_labels" not in self.__dict__:
            self.score_labels = [None for i in range(NUM_VEHICLES)]
        if "transforms" not in self.__dict__:
            self.transforms = [None for i in range(NUM_VEHICLES)]        

        # Construct a viewer for this car with score label and transform object
        if self.viewers[car_idx] is None:
            self.viewers[car_idx] = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_labels[car_idx] = pyglet.text.Label('?', font_size=18,
            x=10, y=30,
            anchor_x='left', anchor_y='center',
            color=(255,255,255,255))
            self.transforms[car_idx] = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

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
        if self.last_pid[car_idx] != -1:
            relevant_node = self.last_relevant_node[car_idx]
            classification_val = self.loc[car_idx] == "right"
            if classification_val:
                shift_pos = self.off_params
                shift_pos = (
                    shift_pos[1]+relevant_node[1]*EDGE_WIDTH,
                    shift_pos[0]+relevant_node[0]*EDGE_WIDTH
                )
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
        rni, rnj = self.last_relevant_node[car_idx]
        neighbor_info = list(self.lattice[rni, rnj, 1:])
        my_last_rect_pid = self.last_rect_pid[car_idx]
        only_turn = None
        if my_last_rect_pid and my_last_rect_pid != -1:
            my_dir_last = self.directions[my_last_rect_pid]
            was_horizontal = my_dir_last in ["l", "r"]
            if np.sum(neighbor_info) == 2:
                if neighbor_info == [False, False, True, True]:
                    if was_horizontal:
                        only_turn = "right"
                    else:
                        only_turn = "left"
                elif neighbor_info == [True, True, False, False]:
                    if was_horizontal:
                        only_turn = "right"
                    else:
                        only_turn = "left"
                elif neighbor_info == [False, True, True, False]:
                    if was_horizontal:
                        only_turn = "left"
                    else:
                        only_turn = "right"
                elif neighbor_info == [True, False, False, True]:
                    if was_horizontal:
                        only_turn = "left"
                    else:
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
            # image_data.save('tmp%d.png'%car_idx)
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
            arr = rgb2gray(arr)
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
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
            for geom in self.viewers[car_idx].onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H, car_idx=car_idx)
            win.flip()

        self.viewers[car_idx].onetime_geoms = []
        return arr, [showed_traffic_light, count_lights if count_lights else None, only_turn]

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
        if self.curr_node is not None:
            self.score_labels[car_idx].text = "%.3f, %s, %.2f [%s]" % (self.tot_reward[car_idx], self.loc[car_idx], true_speed, self.options.HUMAN_NAMES[self.curr_node[car_idx]])
        else:
            self.score_labels[car_idx].text = "%.3f, %s, %.2f" % (self.tot_reward[car_idx], self.loc[car_idx], true_speed)
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
                _, _ = env.render(car_idx=car_idx)

            # if done or restart: break

    # End simulation
    env.close()

