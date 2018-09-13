import numpy as np

dist = lambda a, b, c, d: np.sqrt((a-c)**2+(b-d)**2)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Initially created by Oleg Klimov.
# Licensed on the same terms as the rest of OpenAI Gym.
# Modified by Ashish Gaurav.

STATE_W, STATE_H = 96, 96
VIDEO_W, VIDEO_H = 700, 700
# Comment this out if you're not testing
# Also uncomment the transform part
# WINDOW_W, WINDOW_H = VIDEO_W, VIDEO_H
WINDOW_W, WINDOW_H = 512, 512
GRID_COLS, GRID_ROWS = 4, 4
PROB_EDGE = 0.75
RANDOM_DELETIONS = 6
LATTICE_CONSTRAINTS = [(0, 0), (0, 0), (0, np.inf), (0, 0), (2, np.inf)]

PLAYFIELD   = 800        # Game over boundary
FPS         = 50
ZOOM        = 3          # Camera zoom
ZOOM_FOLLOW = False      # Set to False for fixed view (don't use zoom)
SCALE       = 1.0

LANE_WIDTH = 12
LANE_SEP = 1
EDGE_WIDTH = 150
TRAFFIC_LIGHT_R = LANE_WIDTH//3
TRAFFIC_LIGHT_R2 = 2*LANE_WIDTH//4

NUM_VEHICLES = 1

ConvertRGB = lambda x: list(map(lambda y: y/255.0, x))
TRAFFIC_LIGHT_OUTER_COLOR = ConvertRGB((80, 126, 27))
TRAFFIC_LIGHT_INNER_COLOR = ConvertRGB((255, 255, 255))
ROAD_COLOR = [0.4, 0.4, 0.4]
LANE_SEP_COLOR = [0.6, 0.6, 0.6]

BUFFER_SIZE = 10000
MIN_BUFFER_SIZE_BEFORE_TRAIN = 3000
BATCH_SIZE = 16
GAMMA = 0.95
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic

action_dim = 3  #Steering/Acceleration/Brake
state_dim = (96, 96, 1)

train_indicator = 1 #1 means Train, 0 means simply Run

def isLaneRect(rect):
	xset, yset = set(), set()
	for pt in rect:
		xset.add(pt[0])
		yset.add(pt[1])
	return len(xset) == 2 and len(yset) == 2 and (max(xset)-min(xset) == LANE_WIDTH or max(yset)-min(yset) == LANE_WIDTH)

EXPLORE = 100000.
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.0001