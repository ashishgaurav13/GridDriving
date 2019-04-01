import numpy as np
import random

from grid_driving import GridDriving
from constants import *
from env_utils import *
from pyglet.window import key

# Example file on how to use the environment

# Example lattice
env = GridDriving([
    [1, 1], 
    [1, 1],
    [1, 1]
])

# actions
actions = [np.array( [0.0, 0.0, 0.0] ) for i in range(NUM_VEHICLES)]

def key_press(k, mod, car_idx):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  actions[car_idx][0] = -1.0
    if k==key.RIGHT: actions[car_idx][0] = +1.0
    if k==key.UP:    actions[car_idx][1] = +1.0
    if k==key.DOWN:  actions[car_idx][2] = +0.8   # set 1.0 for wheels to block to zero rotation

def key_release(k, mod, car_idx):
    if k==key.LEFT  and actions[car_idx][0]==-1.0: actions[car_idx][0] = 0
    if k==key.RIGHT and actions[car_idx][0]==+1.0: actions[car_idx][0] = 0
    if k==key.UP:    actions[car_idx][1] = 0
    if k==key.DOWN: actions[car_idx][2] = 0

env.reset()
total_rewards = np.array(make_n_rewards(NUM_VEHICLES))

for car_idx in range(NUM_VEHICLES):
    env.render(car_idx)
    env.viewers[car_idx].window.on_key_press = lambda k, mod: key_press(k, mod, car_idx)
    env.viewers[car_idx].window.on_key_release = lambda k, mod: key_release(k, mod, car_idx)

    
while True:

    states, rewards, done_values, info = env.step(actions)
    total_rewards += np.array(rewards)
    
    # Render a viewer for each car
    for car_idx in range(NUM_VEHICLES):
        _, _ = env.render(car_idx=car_idx)

    # if done or restart: break

# End simulation
env.close()