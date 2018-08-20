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

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

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
