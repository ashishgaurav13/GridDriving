from grid_driving import GridDriving
import numpy as np
import random
import argparse
import math, timeit
import sys

import tensorflow as tf
from keras import backend as K
from utils import initTf
from constants import *

import curses
from curses import wrapper

import warnings
warnings.filterwarnings("ignore")

import datetime # for logging timestamp
import traceback

np.set_printoptions(suppress=True)
initTf(tf, K)

# Create Env
env = GridDriving()

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

while True:

    s_t = env.reset()
    a_t = [np.random.choice(4) for i in range(len(s_t))]

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
    step += 1

    # Render a viewer for each car
    for car_idx in range(NUM_VEHICLES):
        env.render(car_idx=car_idx)
    
    if sum(done) > 0:
        break

    ep += 1

env.end()
