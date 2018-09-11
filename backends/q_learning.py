from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
from keras import backend as K
from keras.initializers import normal, identity
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, merge, Lambda, Conv2D
from .replay_buffer import ReplayBuffer

import sys
sys.path.append('../')
from constants import *

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
        C1 = Conv2D(32, kernel_size=(4, 4), strides=2, init='uniform', activation='relu')(S)
        C2 = Conv2D(64, kernel_size=(3, 3), strides=2, init='uniform', activation='relu')(C1)
        C3 = Conv2D(64, kernel_size=(3, 3), strides=1, init='uniform', activation='relu')(C2)
        F = Flatten()(C3)
        h0 = Dense(192, activation='relu', init='uniform')(F)
        h1 = Dense(512, activation='relu', init='uniform')(h0)
        O = Dense(4, activation='softmax', init='uniform')(h1)
        self.model = Model(input=S, output=O)
        self.adam = Adam(lr=1e-5)
        self.model.compile(loss='mse',optimizer=self.adam)
        self.load_weights()
        self.buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
        # print(self.model.summary())
        print('Created network %d (%d params)' % (self.node, self.model.count_params()))

    def load_weights(self):
        #Now load the weight
        try:
            self.model.load_weights("weights/model%d.h5" % self.node)
            # print("Weights loaded for node %d." % self.node)
        except:
            pass
            # print("Cannot find weights for node %d." % self.node)

    def save_weights(self):
        self.model.save_weights("weights/model%d.h5" % self.node, overwrite=True)
        
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