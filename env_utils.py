import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

def make_n_action_spaces(n):
    return [spaces.Box( 
        low=np.array([-1,0,0]), 
        high=np.array([+1,+1,+1])) for i in range(n)]

def make_n_state_spaces(n, state_dims):
    return [spaces.Box(
        low=0, high=255, 
        shape=state_dims, 
        dtype=np.uint8) for i in range(n)]

def seed(env, seed=None):
    env.np_random, seed = seeding.np_random(seed)
    return [seed]

def make_n_rewards(n, init_value=0.0):
    return [init_value for i in range(n)]

def make_n_done_values(n, init_value=False):
    return [init_value for i in range(n)]

def make_n_times(n, init_value=0.0):
    return [init_value for i in range(n)]
