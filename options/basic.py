import sys
sys.path.append('../')

from constants import *
from backends.ddpg import DDPGPolicyNode
from backends.q_learning import QPolicyNode
from copy import deepcopy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class BasicOptions:

    def __init__(self):

        self.drive_straight = 1
        self.decelerate_to_halt = 2
        self.stop_and_wait3 = [3, 5]
        self.stop_and_wait4 = [3, 4, 5]
        self.left_on_junc = 6
        self.straight_on_junc = 7
        self.right_on_junc = 8
        self.off_road = 9

        self.HUMAN_NAMES = {}
        self.HUMAN_NAMES[1] = 'drive_fwd'
        self.HUMAN_NAMES[2] = 'decel_to_halt'
        self.HUMAN_NAMES[3] = 'wait_left'
        self.HUMAN_NAMES[4] = 'wait_straight'
        self.HUMAN_NAMES[5] = 'wait_right'
        self.HUMAN_NAMES[6] = 'left_on_junc'
        self.HUMAN_NAMES[7] = 'fwd_on_junc'
        self.HUMAN_NAMES[8] = 'right_on_junc'
        self.HUMAN_NAMES[9] = 'off_road'

        self.ou_params_fast = [0, 0.6, 0.3, 2, 0.4, 0.1, 2, 0.4, 0.1]
        self.ou_params_slow = [0, 0.6, 0.3, 1.33, 0.3, 0.1, 1.33, 0.3, 0.1]

        self.policies = [ \
            None,
            QPolicyNode(1, self.init1, self.exit1),
            QPolicyNode(2, self.init2, self.exit2),
            QPolicyNode(3, self.init3, self.exit3),
            QPolicyNode(4, self.init4, self.exit4),
            QPolicyNode(5, self.init5, self.exit5),
            QPolicyNode(6, self.init678, self.exit678),
            QPolicyNode(7, self.init678, self.exit678),
            QPolicyNode(8, self.init678, self.exit678),
            QPolicyNode(9, self.init9, self.exit9)
        ]

        self.lc_node1 = [[0, 0, 0] for i in range(NUM_VEHICLES)] # local counts for node 1
        self.pos_node1 = [[] for i in range(NUM_VEHICLES)]
        self.last_rect_node1 = [set() for i in range(NUM_VEHICLES)]
        self.assign_node1 = [False for i in range(NUM_VEHICLES)]
        self.entering_node1 = [True for i in range(NUM_VEHICLES)]

        self.start_vel_node2 = [0 for i in range(NUM_VEHICLES)]
        self.end_vel_node2 = [0 for i in range(NUM_VEHICLES)]
        self.assign_node2 = [False for i in range(NUM_VEHICLES)]

        self.num_brakes_node3 = [0 for i in range(NUM_VEHICLES)]
        self.assign_node3 = [False for i in range(NUM_VEHICLES)]

        self.num_brakes_node4 = [0 for i in range(NUM_VEHICLES)]
        self.assign_node4 = [False for i in range(NUM_VEHICLES)]

        self.num_brakes_node5 = [0 for i in range(NUM_VEHICLES)]
        self.assign_node5 = [False for i in range(NUM_VEHICLES)]

        self.target_pos_node6 = [(0, 0) for i in range(NUM_VEHICLES)]
        self.entering_node6 = [False for i in range(NUM_VEHICLES)]
        self.assign_node6 = [False for i in range(NUM_VEHICLES)]

        self.target_pos_node7 = [(0, 0) for i in range(NUM_VEHICLES)]
        self.entering_node7 = [False for i in range(NUM_VEHICLES)]
        self.assign_node7 = [False for i in range(NUM_VEHICLES)]

        self.target_pos_node8 = [(0, 0) for i in range(NUM_VEHICLES)]
        self.entering_node8 = [False for i in range(NUM_VEHICLES)]
        self.assign_node8 = [False for i in range(NUM_VEHICLES)]

        self.entering_node9 = [False for i in range(NUM_VEHICLES)]
        self.last_node_before_9 = [0 for i in range(NUM_VEHICLES)]

    # Policy 1: learn to go right lane on a rectangle
    def init1(self, info, car_idx):
        result = info['on_rect'] and info['traffic_light'] is None and not info['off_road']
        return result

    def exit1(self, info, car_idx):
        if info['speed'] >= 10:
            if not info['off_road']:
                self.pos_node1[car_idx].append(np.array([info['pos'].x, info['pos'].y]))
            print(info['last_rect'])
            if Polygon(info['last_rect']).contains(Point(info['pos'].x, info['pos'].y)):
                self.last_rect_node1[car_idx].add(deepcopy(info['last_rect']))
            if info['lane_localization'] == "left":
                self.lc_node1[car_idx][0] += 1
            elif info['lane_localization'] == "right":
                self.lc_node1[car_idx][1] += 1
            else:
                self.lc_node1[car_idx][2] += 1

        if info['only_turn'] == "left" and info['junction']:
            return [self.off_road, self.left_on_junc]
        elif info['only_turn'] == "right" and info['junction']:
            return [self.off_road, self.right_on_junc]
        else:
            return [self.off_road, self.decelerate_to_halt]

    # Policy 2: decelerate to halt to the traffic light
    def init2(self, info, car_idx):
        result = info['on_rect'] and info['traffic_light'] is not None and not info['off_road']
        if result:
            self.start_vel_node2[car_idx] = info['speed']
        return result

    def exit2(self, info, car_idx):
        type_intersection = info['type_intersection']
        if type_intersection is not None:
            # print(type_intersection)
            if type_intersection == 3:
                next_turn = np.random.choice(self.stop_and_wait3, 1)[0]
            elif type_intersection == 4:
                next_turn = np.random.choice(self.stop_and_wait4, 1)[0]
            return [self.off_road, next_turn]
        else:
            return [self.off_road]

    # Policy 3: halt until we have left sign
    def init3(self, info, car_idx):
        return info['traffic_light'] is not None and not info['off_road']

    def exit3(self, info, car_idx):
        if info['traffic_light'] == 'left':
            return [self.off_road, self.left_on_junc]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.left_on_junc]
        else:
            return [self.off_road]

    # Policy 4: halt until we have front sign
    def init4(self, info, car_idx):
        return info['traffic_light'] is not None and not info['off_road']

    def exit4(self, info, car_idx):
        if info['traffic_light'] == 'straight':
            return [self.off_road, self.straight_on_junc]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.straight_on_junc]
        else:
            return [self.off_road]

    # Policy 5: halt until we have right sign
    def init5(self, info, car_idx):
        return info['traffic_light'] is not None and not info['off_road']

    def exit5(self, info, car_idx):
        if info['traffic_light'] == 'right':
            return [self.off_road, self.right_on_junc]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.right_on_junc]
        else:
            return [self.off_road]

    # Policy 6, 7, 8: make a left turn, straight, right turn
    def init678(self, info, car_idx):
        return info['junction'] and not info['off_road']

    def exit678(self, info, car_idx):
        return [self.off_road, self.drive_straight]

    # Policy 9: if we're offroad
    def init9(self, info, car_idx):
        return info['off_road']

    def exit9(self, info, car_idx):
        return [1, 2, 6, 7, 8]