import sys, math
import numpy as np
import random

# Lattice is a set of h x w points, randomly set or unset. We also 
# encode neighbor information. Overall the dimensionality is
# h x w x 5, where (i, j, 0) denotes value of point (i, j). while
# (i, j, 1:4) denote values of neighbors going top, right, bottom, left.
def construct_lattice(h, w):
	lattice = np.zeros(shape=(h, w, 5), dtype=bool)
	for i in range(h):
		for j in range(w):
			lattice[i][j][0] = bool(random.getrandbits(1))
			# update top neighbor
			if i-1 >= 0:
				lattice[i-1][j][3] = lattice[i][j][0]
			# update right neighbor
			if j+1 < w:
				lattice[i][j+1][4] = lattice[i][j][0]
			# update bottom neighbor
			if i+1 < h:
				lattice[i+1][j][1] = lattice[i][j][0]
			# update left neighbor
			if j-1 >= 0:
				lattice[i][j-1][2] = lattice[i][j][0]
	return lattice

# given center (cx, cy) and radius r, find a set of points
# that approximates the arc between angle A and angle B
# where B >= A (degrees); find num_points number of points
def arc_points(A, B, cx, cy, r, num_points):
	pts = []
	A, B = A*math.pi/180.0, B*math.pi/180.0
	diff = B-A
	step = diff/(num_points-1)
	for div in range(num_points):
		pts.append((cx+r*math.cos(A+step*div), cy+r*math.sin(A+step*div)))
	return pts

def translate(pt, xs, ys):
	return (pt[0]+xs, pt[1]+ys)

def rect(x1, x2, y1, y2):
	rects = []
	rects.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
	return rects

def outer_arc(cx, cy, r, A, B):
	arc_pts = arc_points(A, B, cx, cy, r, num_points=5)
	final_pts = []
	for si in range(num_points-1):
		# choose sector
		final_pts.append([(cx, cy), arc_pts[si], arc_pts[si+1]])
	return final_pts

# B-A = 90 degrees
def inner_arc(cx, cy, r, A, B, fake_x, fake_y):
	arc_pts = arc_points(A, B, fake_x, fake_y, r, num_points=3)


# Given 5 values for a point (1 for itself and 4 for neighbors)
# construct polygons needed to draw this out; Lane width is lane_w
# and from this point draw only until distance d
def construct_polygon(i, j, point_info, lane_w, d):
	neighbors = list(point_info[1:5])
	polygons = []
	# ---x
	if neighbors == [False, False, False, True]:
		polygons += rect(-d, 0, 0, lane_w)
		polygons += rect(-d, 0, -lane_w, 0)
		polygons += sectors(-90, 90, )