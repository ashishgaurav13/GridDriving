import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from .replay_buffer import ReplayBuffer

import sys
sys.path.append('../')
from constants import *

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class OU(object):

    def function(self, x, mu, theta, sigma):
        # print('got %f, mu=%f, theta=%f, sigma=%f' % (x, mu, theta, sigma))
        return theta * (mu - x) + sigma * np.random.randn(1)

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=state_size)  
        C = Conv2D(16, kernel_size=(3, 3), init='uniform', activation='relu')(S)
        F = Flatten()(C)
        h0 = Dense(HIDDEN1_UNITS, activation='relu', init='uniform')(F)
        h1 = Dense(HIDDEN2_UNITS, activation='relu', init='uniform')(h0)
        Steering = Dense(1, activation='tanh', init='uniform')(h1)  
        Acceleration = Dense(1, activation='sigmoid', init='uniform')(h1)   
        Brake = Dense(1, activation='sigmoid', init='uniform')(h1) 
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=state_size)
        C = Conv2D(16, kernel_size=(3, 3), init='uniform', activation='relu')(S)
        F = Flatten()(C)

        A = Input(shape=[action_dim],name='action2')   

        w1 = Dense(HIDDEN1_UNITS, activation='relu', init='uniform')(F)
        a1 = Dense(HIDDEN2_UNITS, activation='linear', init='uniform')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear', init='uniform')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu', init='uniform')(h2)
        V = Dense(action_dim,activation='linear', init='uniform')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

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
        self.OU = OU()       #Ornstein-Uhlenbeck Process
        self.make_network()

    def make_network(self):
        self.actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
        self.load_weights()
        # print('Created network %d' % self.node)

    def load_weights(self):
        #Now load the weight
        try:
            self.actor.model.load_weights("weights/actormodel%d.h5" % self.node)
            self.critic.model.load_weights("weights/criticmodel%d.h5" % self.node)
            self.actor.target_model.load_weights("weights/actormodel%d.h5" % self.node)
            self.critic.target_model.load_weights("weights/criticmodel%d.h5" % self.node)
            # print("Weights loaded for node %d." % self.node)
        except:
            pass
            # print("Cannot find weights for node %d." % self.node)

    def save_weights(self):
        self.actor.model.save_weights("weights/actormodel%d.h5" % self.node, overwrite=True)
        # with open("actormodel.json", "w") as outfile:
        #     json.dump(actor.model.to_json(), outfile)

        self.critic.model.save_weights("weights/criticmodel%d.h5" % self.node, overwrite=True)
        # with open("criticmodel.json", "w") as outfile:
        #     json.dump(critic.model.to_json(), outfile)


    # predict action given state
    def actor_predict(self, state, train_indicator=0):
        a_t = np.zeros([1, action_dim])
        noise_t = np.zeros([1, action_dim])
        a_t_original = self.actor.model.predict(state.reshape(1, *state.shape))
        if self.constants:
            noise_t[0][0] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][0],  self.constants[0] , self.constants[1], self.constants[2])
            noise_t[0][1] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][1],  self.constants[3], self.constants[4], self.constants[5])
            noise_t[0][2] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][2], self.constants[6] , self.constants[7], self.constants[8])
        else:
            noise_t[0][0] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][0],  0.0 , 0.6, 0.3)
            noise_t[0][1] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][1],  0.5, 1.0, 0.1)
            noise_t[0][2] = train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][2], -0.1 , 1.0, 0.05)
        # a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        # a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
        a_t[0][0] = noise_t[0][0]
        a_t[0][1] = noise_t[0][1]
        a_t[0][2] = noise_t[0][2]
        # print('orig was %s, new is %s' % (a_t_original[0], a_t[0]))
        
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