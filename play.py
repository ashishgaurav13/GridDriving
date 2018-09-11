from car_grid_driving import CarGridDriving
import numpy as np
import random
import argparse
import math, timeit
import sys

import tensorflow as tf
from keras import backend as K
from utils import initTf, Tee
from options.basic import BasicOptions
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

    options = BasicOptions()

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
        curr_node = [options.drive_straight for i in range(NUM_VEHICLES)]
        s_t = ob

        total_reward = [0. in range(NUM_VEHICLES)]
        counts = [{} for i in range(NUM_VEHICLES)]
        break_episode = False
        for j in range(max_steps):
            loss = [0 for i in range(NUM_VEHICLES)]
            for i in range(NUM_VEHICLES):
                options.policies[curr_node[i]].epsilon -= 1.0 / EXPLORE
            a_t = []
            for i, state in enumerate(s_t):
                greedy_action = np.argmax(options.policies[curr_node[i]].predict(state))
                random_action = np.random.choice(4)
                if np.random.rand(1) <= options.policies[curr_node[i]].epsilon:
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
                options.policies[curr_node[i]].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                if curr_node[i] not in counts[i]:
                    counts[i][curr_node[i]] = 1
                else:
                    counts[i][curr_node[i]] += 1
                if counts[i][curr_node[i]] > 1000:
                    break_episode = True
                if 9 in counts[i] and counts[i][9] > 3:
                    break_episode = True
            
                #Do the batch update
                if train_indicator and options.policies[curr_node[i]].buff.count() >= MIN_BUFFER_SIZE_BEFORE_TRAIN:
                    if j > 50:
                        loss[i] += options.policies[curr_node[i]].train(BATCH_SIZE)

            total_reward = list(np.add(total_reward, r_t))
            s_t = s_t1

            # Change nodes
            for i in range(NUM_VEHICLES):
                for exit_node in options.policies[curr_node[i]].exit(info[i]):
                    if options.policies[exit_node].init(info[i]):
                        curr_node[i] = exit_node

            print("%s\n\nEpisode: %d\nStep: %d\nAction: %s\nReward: %s\nLoss: %s\n\n%s" % (str([options.HUMAN_NAMES[x] for x in curr_node]), ep, step, a_t, r_t, loss, total_reward_str))
            log_file2.write("Episode: %d, Step: %d, Action: %s, Reward: %s, Loss: %s, Currnodes: %s\n" % (ep, step, a_t, r_t, loss, str([options.HUMAN_NAMES[x] for x in curr_node])))
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
                        options.policies[j].save_weights()

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