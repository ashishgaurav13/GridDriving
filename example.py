import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from grid_driving import GridDriving, get_costmap
from constants import *
from env_utils import *
from pyglet.window import key
import matplotlib.pyplot as plt

# Example file on how to use the environment

# Example lattice
env = GridDriving([
    [1, 1], 
    [1, 1],
    [1, 1]
])

# actions
actions = [np.array( [0.0, 0.0, 0.0] ) for i in range(NUM_VEHICLES)]

state = None
info = None

show_costmap = False

# plot costmap on plt
# not ideal but I wasn't sure how to use with render
def plot_costmap():
    costmap = 0.002*state + get_costmap(info)
    plt.imshow(costmap)
    plt.show()
    plt.close()
    
def key_press(k, mod, car_idx):
    if k==key.LEFT:  actions[car_idx][0] = -1.0
    if k==key.RIGHT: actions[car_idx][0] = +1.0
    if k==key.UP:    actions[car_idx][1] = +1.0
    if k==key.DOWN:  actions[car_idx][2] = +0.8   # set 1.0 for wheels to block to zero rotation
    if k==key.SPACE: plot_costmap()
        
        
def key_release(k, mod, car_idx):
    if k==key.LEFT  and actions[car_idx][0]==-1.0: actions[car_idx][0] = 0
    if k==key.RIGHT and actions[car_idx][0]==+1.0: actions[car_idx][0] = 0
    if k==key.UP:    actions[car_idx][1] = 0
    if k==key.DOWN: actions[car_idx][2] = 0
    if k==key.SPACE:
        show_costmap = False

env.reset()
total_rewards = np.array(make_n_rewards(NUM_VEHICLES))
env.viewers[0].window.on_key_press = lambda k, mod: key_press(k, mod, 0)
env.viewers[0].window.on_key_release = lambda k, mod: key_release(k, mod, 0)
env.viewers[1].window.set_visible(False)

while True:

    states, rewards, done_values, info = env.step(actions)
    total_rewards += np.array(rewards)
    # Render a viewer for each car
    state = np.squeeze(states)
    env.render()

    if done_values: break

# End simulation
env.close()
