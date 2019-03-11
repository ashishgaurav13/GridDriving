import numpy as np
from pyglet.window import key

# map integer keypress to action
def key_press(k):
    action = [0.0, 0.0, 0.0]
    if k==key.LEFT:  action[0] = -1.0
    if k==key.RIGHT:  action[0] = +1.0
    if k==key.UP:  action[1] = +1.0
    if k==key.DOWN:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    return action

def get_control_vec(a):
    if a == 0: return np.add(key_press(key.LEFT), key_press(key.UP))
    elif a == 1: return np.array(key_press(key.UP))
    elif a == 2: return np.add(key_press(key.RIGHT), key_press(key.UP))
    elif a == 3: return np.array(key_press(key.DOWN))