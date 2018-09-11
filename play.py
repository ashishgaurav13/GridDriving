from car_grid_driving import CarGridDriving
import numpy as np
import random
import argparse
import math, timeit
import sys

import tensorflow as tf
from keras import backend as K
from utils import initTf, Tee
from backends.ddpg import DDPGPolicyNode
from backends.q_learning import QPolicyNode
from constants import *

import curses
from curses import wrapper

import warnings
warnings.filterwarnings("ignore")

import datetime # for logging timestamp

initTf(tf, K)

def playGame():    

    global train_indicator
    EXPLORE = 10000.
    episode_count = 100000
    max_steps = 5000
    reward = 0
    done = False
    step = 0

    log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
    log_file_name2 = datetime.datetime.now().strftime("more_%Y_%m_%d_%H_%M_%S.txt")
    log_file = open("logs/"+log_file_name, "w")
    log_file2 = open("logs/"+log_file_name2, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    drive_straight = 1
    decelerate_to_halt = 2
    stop_and_wait3 = [3, 5]
    stop_and_wait4 = [3, 4, 5]
    left_on_junc = 6
    straight_on_junc = 7
    right_on_junc = 8
    off_road = 9

    HUMAN_NAMES = {}
    HUMAN_NAMES[1] = 'drive_straight'
    HUMAN_NAMES[2] = 'decelerate_to_halt'
    HUMAN_NAMES[3] = 'wait_left'
    HUMAN_NAMES[4] = 'wait_straight'
    HUMAN_NAMES[5] = 'wait_right'
    HUMAN_NAMES[6] = 'left_on_junc'
    HUMAN_NAMES[7] = 'straight_on_junc'
    HUMAN_NAMES[8] = 'right_on_junc'
    HUMAN_NAMES[9] = 'off_road'

    ou_params_fast = [0, 0.6, 0.3, 2, 0.4, 0.1, 2, 0.4, 0.1]
    ou_params_slow = [0, 0.6, 0.3, 1.33, 0.3, 0.1, 1.33, 0.3, 0.1]

    # Policy 1: learn to go right lane on a rectangle
    def init1(info):
        return info['on_rect'] and info['traffic_light'] is None and not info['off_road']
    def exit1(info):
        if info['only_turn'] == "left" and info['junction']:
            return [off_road, left_on_junc, decelerate_to_halt]
        elif info['only_turn'] == "right" and info['junction']:
            return [off_road, right_on_junc, decelerate_to_halt]
        else:
            return [off_road, decelerate_to_halt]
    # policy1 = PolicyNode(1, init1, exit1, ou_params_fast)
    policy1 = QPolicyNode(1, init1, exit1)

    # Policy 2: decelerate to halt to the traffic light
    def init2(info):
        return info['on_rect'] and info['traffic_light'] is not None and not info['off_road']
    def exit2(info):
        type_intersection = info['type_intersection']
        if type_intersection is not None:
            # print(type_intersection)
            if type_intersection == 3:
                next_turn = np.random.choice(stop_and_wait3, 1)[0]
            elif type_intersection == 4:
                next_turn = np.random.choice(stop_and_wait4, 1)[0]
            return [off_road, next_turn]
        else:
            return [off_road]
    # policy2 = PolicyNode(2, init2, exit2, ou_params_slow)
    policy2 = QPolicyNode(2, init2, exit2)

    # Policy 3: halt until we have left sign
    def init3(info):
        return info['traffic_light'] is not None and not info['off_road']
    def exit3(info):
        if info['traffic_light'] == 'left':
            return [off_road, left_on_junc, drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [off_road, left_on_junc, drive_straight]
        else:
            return [off_road, drive_straight]
    # policy3 = PolicyNode(3, init3, exit3, ou_params_slow)
    policy3 = QPolicyNode(3, init3, exit3)

    # Policy 4: halt until we have front sign
    def init4(info):
        return info['traffic_light'] is not None and not info['off_road']
    def exit4(info):
        if info['traffic_light'] == 'straight':
            return [off_road, straight_on_junc, drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [off_road, straight_on_junc, drive_straight]
        else:
            return [off_road, drive_straight]
    # policy4 = PolicyNode(4, init4, exit4, ou_params_slow)
    policy4 = QPolicyNode(4, init4, exit4)

    # Policy 5: halt until we have right sign
    def init5(info):
        return info['traffic_light'] is not None and not info['off_road']
    def exit5(info):
        if info['traffic_light'] == 'right':
            return [off_road, right_on_junc, drive_straight]
        elif info['traffic_light'] is None:
            # we accidentally entered this state, no traffic light now, just move
            return [off_road, right_on_junc, drive_straight]
        else:
            return [off_road, drive_straight]
    # policy5 = PolicyNode(5, init5, exit5, ou_params_slow)
    policy5 = QPolicyNode(5, init5, exit5)

    # Policy 6, 7, 8: make a left turn, straight, right turn
    def init678(info):
        return info['junction'] and not info['off_road']
    def exit678(info):
        return [off_road, drive_straight]
    # policy6 = PolicyNode(6, init678, exit678, ou_params_fast)
    # policy7 = PolicyNode(7, init678, exit678, ou_params_fast)
    # policy8 = PolicyNode(8, init678, exit678, ou_params_fast)
    policy6 = QPolicyNode(6, init678, exit678)
    policy7 = QPolicyNode(7, init678, exit678)
    policy8 = QPolicyNode(8, init678, exit678)

    # Policy 9: if we're offroad
    def init9(info):
        return info['off_road']
    def exit9(info):
        return [1, 2, 6, 7, 8]
    # policy9 = PolicyNode(9, init9, exit9, ou_params_slow)
    policy9 = QPolicyNode(9, init9, exit9)

    # Put all together
    policies = [None, policy1, policy2, policy3, policy4, policy5, policy6, policy7, policy8, policy9]

    # Create Env
    env = CarGridDriving()

    # map integer keypress to action
    def key_press(k):
        action = [0.0, 0.0, 0.0]
        if k==1:  action[0] = -1.0
        if k==2:  action[0] = +1.0
        if k==3:  action[1] = +1.0
        if k==4:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        return action

    def get_control_vec(a):
        if a == 0: return np.add(key_press(1), key_press(3))
        elif a == 1: return np.array(key_press(3))
        elif a == 2: return np.add(key_press(2), key_press(3))
        elif a == 3: return np.array(key_press(4))

    ep = 1
    total_reward_str = ""
    while ep <= episode_count:

        print("Episode : " + str(ep))
        log_file2.write("Episode : " + str(ep) + "\n")

        ob = env.reset()
        curr_node = [drive_straight for i in range(NUM_VEHICLES)]
        s_t = ob

        total_reward = [0. in range(NUM_VEHICLES)]
        counts = [{} for i in range(NUM_VEHICLES)]
        break_episode = False
        for j in range(max_steps):
            loss = [0 for i in range(NUM_VEHICLES)]
            for i in range(NUM_VEHICLES):
                policies[curr_node[i]].epsilon -= 1.0 / EXPLORE
            a_t = []
            for i, state in enumerate(s_t):
                # a_t.append(policies[curr_node[i]].actor_predict(state, train_indicator))
                greedy_action = np.argmax(policies[curr_node[i]].predict(state))
                random_action = np.random.choice(4)
                if np.random.rand(1) <= policies[curr_node[i]].epsilon:
                    a_t.append(random_action)
                else:
                    a_t.append(greedy_action)

            # Info has "traffic"
            loop = True
            while loop:
                try:
                    ob, r_t, done, info = env.step([get_control_vec(e) for e in a_t], curr_node)
                except:
                    loop = True
                    print('wtf')
                else:
                    loop = False

            s_t1 = ob
        
            for i in range(NUM_VEHICLES):
                policies[curr_node[i]].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                if curr_node[i] not in counts[i]:
                    counts[i][curr_node[i]] = 1
                else:
                    counts[i][curr_node[i]] += 1
                if counts[i][curr_node[i]] > 1000:
                    break_episode = True
                if 9 in counts[i] and counts[i][9] > 3:
                    break_episode = True
            
                #Do the batch update
                if train_indicator and policies[curr_node[i]].buff.count() >= MIN_BUFFER_SIZE_BEFORE_TRAIN:
                    if j > 50:
                        loss[i] += policies[curr_node[i]].train(BATCH_SIZE)

            total_reward = list(np.add(total_reward, r_t))
            s_t = s_t1

            # Change nodes
            for i in range(NUM_VEHICLES):
                for exit_node in policies[curr_node[i]].exit(info[i]):
                    if policies[exit_node].init(info[i]):
                        curr_node[i] = exit_node

            print("%s\n\nEpisode: %d\nStep: %d\nAction: %s\nReward: %s\nLoss: %s\n\n%s" % (str([HUMAN_NAMES[x] for x in curr_node]), ep, step, a_t, r_t, loss, total_reward_str))
            log_file2.write("Episode: %d, Step: %d, Action: %s, Reward: %s, Loss: %s, Currnodes: %s\n" % (ep, step, a_t, r_t, loss, str([HUMAN_NAMES[x] for x in curr_node])))            
            log_file2.write("All: %s\n" % (info))
            step += 1

            # Render a viewer for each car
            for car_idx in range(NUM_VEHICLES):
                env.render(car_idx=car_idx)
            
            if break_episode or sum(done) > 0:
                break

        if np.mod(ep, 3) == 0:
            if train_indicator:
                for i in range(NUM_VEHICLES):
                    for j in range(1, len(HUMAN_NAMES)+1):
                        policies[j].save_weights()

        total_reward_str = str(total_reward)
        print("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        log_file2.write("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward) + "\n")
        log_file2.write("Total Step: " + str(step) + "\n")
        log_file2.write("\n")
        log_file2.flush()

        ep += 1

    env.end()  # This is for shutting down TORCS
    log_file2.write("Finish.\n")
    log_file2.close()

if __name__ == "__main__":
    playGame()