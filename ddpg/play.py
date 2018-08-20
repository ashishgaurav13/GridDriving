import sys
sys.path.append('../')
from car_grid_driving import CarGridDriving
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
from keras import backend as K

import math
from keras.initializers import normal, identity
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, merge, Lambda, Conv2D

from ReplayBuffer import ReplayBuffer
from actor_critic import ActorNetwork, CriticNetwork
import timeit

import warnings
warnings.filterwarnings("ignore")

import datetime # for logging timestamp

NUM_VEHICLES = 1

BUFFER_SIZE = 10000
MIN_BUFFER_SIZE_BEFORE_TRAIN = 1000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic

action_dim = 3  #Steering/Acceleration/Brake
state_dim = (96, 96, 1)

class OU(object):

    def function(self, x, mu, theta, sigma):
        print('got %f, mu=%f, theta=%f, sigma=%f' % (x, mu, theta, sigma))
        return theta * (mu - x) + sigma * np.random.randn(1)

OU = OU()       #Ornstein-Uhlenbeck Process

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Construct a policy node which has a Q network with
# init, exit conditions
class QPolicyNode:

    # node: the node number
    #
    # init: function that takes info, returns true or false
    #
    # exit: exit nodes possible, just check their init
    #
    def __init__(self, node, init, exit):
        self.node = node
        self.init = init
        self.exit = exit
        self.epsilon = 1
        self.make_network()

    def make_network(self):
        S = Input(shape=state_dim)
        C = Conv2D(32, kernel_size=(3, 3), init='uniform', activation='relu')(S)
        F = Flatten()(C)
        h0 = Dense(256, activation='relu', init='uniform')(F)
        h1 = Dense(512, activation='relu', init='uniform')(h0)
        O = Dense(4, activation='softmax', init='uniform')(h1)
        self.model = Model(input=S, output=O)
        self.adam = Adam(lr=1e-5)
        self.model.compile(loss='mse',optimizer=self.adam)
        self.load_weights()
        self.buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
        print('Created network %d' % self.node)

    def load_weights(self):
        #Now load the weight
        try:
            self.model.load_weights("model%d.h5" % self.node)
            # print("Weights loaded for node %d." % self.node)
        except:
            pass
            # print("Cannot find weights for node %d." % self.node)

    def save_weights(self):
        self.model.save_weights("model%d.h5" % self.node, overwrite=True)
        
    # predict action given state
    def predict(self, state):
        return self.model.predict(state.reshape(1, *state.shape))[0]

    def add_to_replay(self, state, action, reward, new_state, done):
        self.buff.add(state, action, reward, new_state, done)

    def train(self, batch_size):
        batch = self.buff.getBatch(batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        
        y_t = self.model.predict(states)
        Q_sa = self.model.predict(new_states)
           
        for k in range(len(batch)):
            if dones[k]:
                y_t[k, actions[k]] = rewards[k]
            else:
                y_t[k, actions[k]] = rewards[k] + GAMMA*np.max(Q_sa[k])
       
        loss = self.model.train_on_batch(states, y_t)
        return loss


# Construct a policy node which has a DDPG network with
# init, exit conditions
class DDPGPolicyNode:

    # node: the node number
    #
    # init: function that takes info, returns true or false
    #
    # exit: exit nodes possible, just check their init
    #
    def __init__(self, node, init, exit, constants=None):
        self.node = node
        self.init = init
        self.exit = exit
        self.epsilon = 1
        self.constants = constants
        self.make_network()

    def make_network(self):
        self.actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
        self.load_weights()
        print('Created network %d' % self.node)

    def load_weights(self):
        #Now load the weight
        try:
            self.actor.model.load_weights("actormodel%d.h5" % self.node)
            self.critic.model.load_weights("criticmodel%d.h5" % self.node)
            self.actor.target_model.load_weights("actormodel%d.h5" % self.node)
            self.critic.target_model.load_weights("criticmodel%d.h5" % self.node)
            # print("Weights loaded for node %d." % self.node)
        except:
            pass
            # print("Cannot find weights for node %d." % self.node)

    def save_weights(self):
        self.actor.model.save_weights("actormodel%d.h5" % self.node, overwrite=True)
        # with open("actormodel.json", "w") as outfile:
        #     json.dump(actor.model.to_json(), outfile)

        self.critic.model.save_weights("criticmodel%d.h5" % self.node, overwrite=True)
        # with open("criticmodel.json", "w") as outfile:
        #     json.dump(critic.model.to_json(), outfile)


    # predict action given state
    def actor_predict(self, state, train_indicator=0):
        a_t = np.zeros([1, action_dim])
        noise_t = np.zeros([1, action_dim])
        a_t_original = self.actor.model.predict(state.reshape(1, *state.shape))
        if self.constants:
            noise_t[0][0] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][0],  self.constants[0] , self.constants[1], self.constants[2])
            noise_t[0][1] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][1],  self.constants[3], self.constants[4], self.constants[5])
            noise_t[0][2] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][2], self.constants[6] , self.constants[7], self.constants[8])
        else:
            noise_t[0][0] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.6, 0.3)
            noise_t[0][1] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][1],  0.5, 1.0, 0.1)
            noise_t[0][2] = train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.0, 0.05)
        # a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        # a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
        a_t[0][0] = noise_t[0][0]
        a_t[0][1] = noise_t[0][1]
        a_t[0][2] = noise_t[0][2]
        print('orig was %s, new is %s' % (a_t_original[0], a_t[0]))
        
        return a_t[0]

    def add_to_replay(self, state, action, reward, new_state, done):
        self.buff.add(state, action, reward, new_state, done)

    def train(self, batch_size):
        batch = self.buff.getBatch(batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
           
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
        loss = self.critic.model.train_on_batch([states,actions], y_t) 
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        return loss


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    EXPLORE = 10000.
    episode_count = 100000
    max_steps = 5000
    reward = 0
    done = False
    step = 0

    log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
    log_file_name2 = datetime.datetime.now().strftime("more_%Y_%m_%d_%H_%M_%S.txt")
    log_file = open(log_file_name, "w")
    log_file2 = open(log_file_name2, "w")
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
            print(type_intersection)
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

    print('Starting ...')
    for ep in range(episode_count):

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

            print("%s\t Episode: %d, Step: %d, Action: %s, Reward: %s, Loss: %s" % (str([HUMAN_NAMES[x] for x in curr_node]), ep, step, a_t, r_t, loss))
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

        print("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        log_file2.write("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward) + "\n")
        log_file2.write("Total Step: " + str(step) + "\n")
        log_file2.write("\n")
        log_file2.flush()

    env.end()  # This is for shutting down TORCS
    log_file2.write("Finish.\n")
    log_file2.close()


if __name__ == "__main__":
    playGame(1)
