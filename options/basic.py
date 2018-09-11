import sys
sys.path.append('../')

from constants import *
from backends.ddpg import DDPGPolicyNode
from backends.q_learning import QPolicyNode

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
        self.HUMAN_NAMES[1] = 'drive_straight'
        self.HUMAN_NAMES[2] = 'decelerate_to_halt'
        self.HUMAN_NAMES[3] = 'wait_left'
        self.HUMAN_NAMES[4] = 'wait_straight'
        self.HUMAN_NAMES[5] = 'wait_right'
        self.HUMAN_NAMES[6] = 'left_on_junc'
        self.HUMAN_NAMES[7] = 'straight_on_junc'
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

    # Policy 1: learn to go right lane on a rectangle
    def init1(self, info):
        return info['on_rect'] and info['traffic_light'] is None and not info['off_road']

    def exit1(self, info):
        if info['only_turn'] == "left" and info['junction']:
            return [self.off_road, self.left_on_junc, self.decelerate_to_halt]
        elif info['only_turn'] == "right" and info['junction']:
            return [self.off_road, self.right_on_junc, self.decelerate_to_halt]
        else:
            return [self.off_road, self.decelerate_to_halt]

    # Policy 2: decelerate to halt to the traffic light
    def init2(self, info):
        return info['on_rect'] and info['traffic_light'] is not None and not info['off_road']

    def exit2(self, info):
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
    def init3(self, info):
        return info['traffic_light'] is not None and not info['off_road']

    def exit3(self, info):
        if info['traffic_light'] == 'left':
            return [self.off_road, self.left_on_junc, self.drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.left_on_junc, self.drive_straight]
        else:
            return [self.off_road, self.drive_straight]
    # Policy 4: halt until we have front sign
    def init4(self, info):
        return info['traffic_light'] is not None and not info['off_road']

    def exit4(self, info):
        if info['traffic_light'] == 'straight':
            return [self.off_road, self.straight_on_junc, self.drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.straight_on_junc, self.drive_straight]
        else:
            return [self.off_road, self.drive_straight]

    # Policy 5: halt until we have right sign
    def init5(self, info):
        return info['traffic_light'] is not None and not info['off_road']

    def exit5(self, info):
        if info['traffic_light'] == 'right':
            return [self.off_road, self.right_on_junc, self.drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [self.off_road, self.right_on_junc, self.drive_straight]
        else:
            return [self.off_road, self.drive_straight]

    # Policy 6, 7, 8: make a left turn, straight, right turn
    def init678(self, info):
        return info['junction'] and not info['off_road']

    def exit678(self, info):
        return [self.off_road, self.drive_straight]

    # Policy 9: if we're offroad
    def init9(self, info):
        return info['off_road']

    def exit9(self, info):
        return [1, 2, 6, 7, 8]