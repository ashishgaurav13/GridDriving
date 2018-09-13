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
import traceback

np.set_printoptions(suppress=True)
initTf(tf, K)

def playGame():    

    global train_indicator
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
    env = CarGridDriving(options)

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
    episode_loss = []
    episode_reward = []
    count_steps = []
    until_last_ep_stats = ""
    q_values_str = ""
    while ep <= episode_count:

        print("Episode : " + str(ep))
        log_file2.write("Episode : " + str(ep) + "\n")

        ob = env.reset()
        curr_node = [options.drive_straight for i in range(NUM_VEHICLES)]
        s_t = ob

        total_reward = [0. in range(NUM_VEHICLES)]
        counts = [{} for i in range(NUM_VEHICLES)]
        break_episode = False

        episode_loss.append([[0 for eli in range(10)] for elvi in range(NUM_VEHICLES)])
        episode_reward.append([[0 for eri in range(10)] for ervi in range(NUM_VEHICLES)])
        count_steps.append([[0 for csi in range(10)] for csvi in range(NUM_VEHICLES)])

        for j in range(max_steps):
            loss = [[0 for k in range(10)] for i in range(NUM_VEHICLES)]
            for i in range(NUM_VEHICLES):
                if options.policies[curr_node[i]].buff.count() >= MIN_BUFFER_SIZE_BEFORE_TRAIN and options.policies[curr_node[i]].epsilon > FINAL_EPSILON:
                    options.policies[curr_node[i]].epsilon -= (INITIAL_EPSILON-FINAL_EPSILON) / EXPLORE
            a_t = []
            for i, state in enumerate(s_t):
                q_values = options.policies[curr_node[i]].predict(state)
                if curr_node[i] == 1 or options.assign_node1[i] == True:
                    q_values_str = str(["%.6f" % item for item in q_values.tolist()])
                else:
                    q_values_str = ""
                greedy_action = np.argmax(q_values)
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
                    traceback.print_exc()
                    exit(1)
                else:
                    loop = False

            s_t1 = ob
        
            for i in range(NUM_VEHICLES):

                # delayed sparse rewards
                if options.assign_node1[i] == True:
                    options.policies[1].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][1] += r_t[i]
                    count_steps[-1][i][1] += 1
                elif options.assign_node2[i] == True:
                    options.policies[2].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][2] += r_t[i]
                    count_steps[-1][i][2] += 1
                elif options.assign_node3[i] == True:
                    options.policies[3].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][3] += r_t[i]
                    count_steps[-1][i][3] += 1
                elif options.assign_node4[i] == True:
                    options.policies[4].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][4] += r_t[i]
                    count_steps[-1][i][4] += 1
                elif options.assign_node5[i] == True:
                    options.policies[5].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][5] += r_t[i]
                    count_steps[-1][i][5] += 1
                elif options.assign_node6[i] == True:
                    options.policies[6].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][6] += r_t[i]
                    count_steps[-1][i][6] += 1
                elif options.assign_node7[i] == True:
                    options.policies[7].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][7] += r_t[i]
                    count_steps[-1][i][7] += 1
                elif options.assign_node8[i] == True:
                    options.policies[8].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][8] += r_t[i]
                    count_steps[-1][i][8] += 1
                elif options.entering_node9[i] == True:
                    options.policies[options.last_node_before_9[i]].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][options.last_node_before_9[i]] += r_t[i]
                    count_steps[-1][i][options.last_node_before_9[i]] += 1
                else:
                    options.policies[curr_node[i]].add_to_replay(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])
                    episode_reward[-1][i][curr_node[i]] += r_t[i]
                    count_steps[-1][i][curr_node[i]] += 1

                if curr_node[i] not in counts[i]:
                    counts[i][curr_node[i]] = 1
                else:
                    counts[i][curr_node[i]] += 1
                if counts[i][curr_node[i]] > 1000:
                    break_episode = True
                if 9 in counts[i] and counts[i][9] > 3:
                    break_episode = True
            
                #Do the batch update
                if train_indicator == 1 and options.policies[curr_node[i]].buff.count() >= MIN_BUFFER_SIZE_BEFORE_TRAIN:
                    # if j > 50:
                    if options.assign_node1[i] == True:
                        loss[i][1] += options.policies[1].train(BATCH_SIZE)
                        episode_loss[-1][i][1] += loss[i][1]
                    elif options.assign_node2[i] == True:
                        loss[i][2] += options.policies[2].train(BATCH_SIZE)
                        episode_loss[-1][i][2] += loss[i][2]
                    elif options.assign_node3[i] == True:
                        loss[i][3] += options.policies[3].train(BATCH_SIZE)
                        episode_loss[-1][i][3] += loss[i][3]
                    elif options.assign_node4[i] == True:
                        loss[i][4] += options.policies[4].train(BATCH_SIZE)
                        episode_loss[-1][i][4] += loss[i][4]
                    elif options.assign_node5[i] == True:
                        loss[i][5] += options.policies[5].train(BATCH_SIZE)
                        episode_loss[-1][i][5] += loss[i][5]
                    elif options.assign_node6[i] == True:
                        loss[i][6] += options.policies[6].train(BATCH_SIZE)
                        episode_loss[-1][i][6] += loss[i][6]
                    elif options.assign_node7[i] == True:
                        loss[i][7] += options.policies[7].train(BATCH_SIZE)
                        episode_loss[-1][i][7] += loss[i][7]
                    elif options.assign_node8[i] == True:
                        loss[i][8] += options.policies[8].train(BATCH_SIZE)
                        episode_loss[-1][i][8] += loss[i][8]
                    elif options.entering_node9[i] == True:
                        loss[i][options.last_node_before_9[i]] += options.policies[options.last_node_before_9[i]].train(BATCH_SIZE)
                        episode_loss[-1][i][options.last_node_before_9[i]] += loss[i][options.last_node_before_9[i]]
                    else:
                        loss[i][curr_node[i]] += options.policies[curr_node[i]].train(BATCH_SIZE)
                        episode_loss[-1][i][curr_node[i]] += loss[i][curr_node[i]]

            total_reward = list(np.add(total_reward, r_t))
            s_t = s_t1

            # Change nodes
            for i in range(NUM_VEHICLES):

                if options.assign_node1[i] == True:
                    options.assign_node1[i] = False
                    options.lc_node1[i] = [0, 0, 0]
                    options.pos_node1[i] = []
                    options.last_rect_node1[i] = set()
                    options.direction_node1[i] = None
                    # exit(0)
                    break_episode = True # TODO
                if options.assign_node2[i] == True:
                    options.start_vel_node2[i] = 0
                    options.end_vel_node2[i] = 0
                    options.assign_node2[i] = False
                if options.assign_node3[i] == True:
                    options.num_brakes_node3[i] = 0
                    options.assign_node3[i] = False
                if options.assign_node4[i] == True:
                    options.num_brakes_node4[i] = 0
                    options.assign_node4[i] = False
                if options.assign_node5[i] == True:
                    options.num_brakes_node5[i] = 0
                    options.assign_node5[i] = False
                if options.assign_node6[i] == True:
                    options.target_pos_node6[i] = (0, 0)
                    options.assign_node6[i] = False
                if options.assign_node7[i] == True:
                    options.target_pos_node7[i] = (0, 0)
                    options.assign_node7[i] = False
                if options.assign_node8[i] == True:
                    options.target_pos_node8[i] = (0, 0)
                    options.assign_node8[i] = False

                if options.entering_node1[i] == True:
                    options.entering_node1[i] = False # TODO: needed?
                if options.entering_node6[i] == True:
                    options.entering_node6[i] = False
                if options.entering_node7[i] == True:
                    options.entering_node7[i] = False
                if options.entering_node8[i] == True:
                    options.entering_node8[i] = False
                if options.entering_node9[i] == True:
                    options.last_node_before_9[i] = 0
                    options.entering_node9[i] = False

                if curr_node[i] == 3:
                    if a_t[i] == 3: # brake
                        options.num_brakes_node3[i] += 1
                if curr_node[i] == 4:
                    if a_t[i] == 3: # brake
                        options.num_brakes_node4[i] += 1
                if curr_node[i] == 5:
                    if a_t[i] == 3: # brake
                        options.num_brakes_node5[i] += 1

                for exit_node in options.policies[curr_node[i]].exit(info[i], i):
                    if options.policies[exit_node].init(info[i], i):
                        if curr_node[i] == 1:
                            options.assign_node1[i] = True
                        if curr_node[i] == 2:
                            options.end_vel_node2[i] = info[i]['speed']
                            options.assign_node2[i] = True
                        if curr_node[i] == 3:
                            options.assign_node3[i] = True
                        if curr_node[i] == 4:
                            options.assign_node4[i] = True
                        if curr_node[i] == 5:
                            options.assign_node5[i] = True
                        if curr_node[i] == 6:
                            options.assign_node6[i] = True
                        if curr_node[i] == 7:
                            options.assign_node7[i] = True
                        if curr_node[i] == 8:
                            options.assign_node8[i] = True

                        if exit_node == 1:
                            options.entering_node1[i] = True
                        if exit_node == 6:
                            options.entering_node6[i] = True
                        if exit_node == 7:
                            options.entering_node7[i] = True
                        if exit_node == 8:
                            options.entering_node8[i] = True
                        if exit_node == 9:
                            options.last_node_before_9[i] = curr_node[i]
                            options.entering_node9[i] = True

                        curr_node[i] = exit_node
                        break # just assign to one exit node (the first one)

            print("Maneuver: %s\nEpisode: %d\nStep: %d\nAction: %s\nReward: %s\nQ: %s\n%s\n\n" % (str([options.HUMAN_NAMES[x] for x in curr_node]), ep, step, a_t, r_t, q_values_str, until_last_ep_stats))
            log_file2.write("Episode: %d, Step: %d, Action: %s, Reward: %s, Loss: %s, Currnodes: %s\n" % (ep, step, a_t, r_t, loss, str([options.HUMAN_NAMES[x] for x in curr_node])))
            log_file2.write("All: %s\n" % (info))
            log_file2.write("EpLoss: %s\n" % (episode_loss))
            log_file2.write("EpReward: %s\n" % (episode_reward))
            # log_file2.write()
            step += 1

            # Render a viewer for each car
            for car_idx in range(NUM_VEHICLES):
                env.render(car_idx=car_idx)
            
            if break_episode or sum(done) > 0:
                break

        if np.mod(ep, 3) == 0:
            if train_indicator:
                for i in range(NUM_VEHICLES):
                    for j in range(1, len(options.HUMAN_NAMES)+1):
                        options.policies[j].save_weights()

        # epblock = episode_reward[-1]
        # for vehindex, vehblock in enumerate(epblock):
        #     for nodeindex, nodeblock in enumerate(vehblock):
        #         cs_respective = count_steps[-1][vehindex][nodeindex]
        #         if cs_respective > 0:
        #             episode_reward[-1][vehindex][nodeindex] /= cs_respective*1.0
        #             episode_loss[-1][vehindex][nodeindex] /= cs_respective*1.0

        tot_reward = np.sum(episode_reward[-10:], axis=0)
        avg_loss = np.mean(episode_loss[-10:], axis=0)
        total_counts = np.sum(count_steps, axis=0)
        until_last_ep_stats = "\noption\t\timm_loss\ttot_rew10\tavg_loss10\tsteps\n"
        for iii in range(1, 10):
            until_last_ep_stats += "%s:\t%.6f\t%.6f\t%.6f\t%d\n" % (options.HUMAN_NAMES[iii], np.array(loss)[:, iii], tot_reward[:, iii], avg_loss[:, iii], total_counts[:, iii])
        # until_last_ep_stats = "\n\nRew: %s\n\nLoss: %s\n" % (episode_reward[-10:], episode_loss[-10:])
        # until_last_ep_stats = "\nAvgRewardUntilLastEp: %s\nAvgLossUntilLastEp: %s\n" % (avg_reward.tolist(), avg_loss.tolist())

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