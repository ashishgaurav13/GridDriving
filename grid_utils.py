import sys, math
import numpy as np
import random
from collections import deque

# BFS to check graph connectivity
def BFS(lattice):
	h, w = lattice.shape
	tovisit = None
	for i in range(h):
		for j in range(w):
			if lattice[i][j]:
				tovisit = deque([(0, 0)])
	if not tovisit: return False # empty graph
	n = np.sum(lattice)
	visited = set()
	while len(tovisit) > 0:
		curr = tovisit.popleft()
		visited.add(curr)
		i, j = curr
		if i-1 >= 0 and (i-1, j) not in visited and lattice[i-1, j]:
			tovisit.append((i-1, j))
		if i+1 < h and (i+1, j) not in visited and lattice[i+1, j]:
			tovisit.append((i+1, j))
		if j-1 >= 0 and (i, j-1) not in visited and lattice[i, j-1]:
			tovisit.append((i, j-1))
		if j+1 < w and (i, j+1) not in visited and lattice[i, j+1]:
			tovisit.append((i, j+1))
	return len(visited) == n

# Lattice is a set of h x w points, randomly set or unset. We also 
# encode neighbor information. Overall the dimensionality is
# h x w x 5, where (i, j, 0) denotes value of point (i, j). while
# (i, j, 1:4) denote values of neighbors going top, right, bottom, left.
def construct_lattice(h, w, p):
	lattice = np.zeros(shape=(h, w, 5), dtype=bool)
	is_connected = False
	while not is_connected:
		for i in range(h):
			for j in range(w):
				lattice[i][j][0] = bool(np.random.choice(2, 1, p=[1-p, p])[0])
		is_connected = BFS(lattice[:, :, 0])
	for i in range(h):
		for j in range(w):
			# update bottom neighbor
			if i-1 >= 0:
				lattice[i-1][j][1] = lattice[i][j][0]
			# update right neighbor
			if j+1 < w:
				lattice[i][j+1][4] = lattice[i][j][0]
			# update top neighbor
			if i+1 < h:
				lattice[i+1][j][3] = lattice[i][j][0]
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

def rect(x1, x2, y1, y2, xpoints=2, ypoints=2):
	rects = []
	xstep = (x2-x1)/(xpoints-1.0)
	ystep = (y2-y1)/(ypoints-1.0)
	for xi in range(xpoints-1):
		for yi in range(ypoints-1):
			left, right = x1+xi*xstep, x1+(xi+1)*xstep
			bottom, top = y1+yi*ystep, y1+(yi+1)*ystep
			rects.append([(left, bottom), (left, top), (right, top), (right, bottom)])
	return rects

def outer_arc(cx, cy, r, A, B, num_points=5):
	arc_pts = arc_points(A, B, cx, cy, r, num_points)
	final_polygons = []
	for si in range(num_points-1):
		# choose sector
		# Hack: having 4 points instead of 3 guarantees that polygon shows up
		final_polygons.append([(cx, cy), (cx, cy), arc_pts[si], arc_pts[si+1]])
	return final_polygons

# B-A = 90 degrees and A, B must be in multiples of 90
# num_points >= 2, y2-y1=x2-x1
def inner_arc(x1, x2, y1, y2, dx, dy, r, A, B, num_points=3):
	arc_pts = arc_points(A, B, dx, dy, r, num_points)
	total_length = x2-x1+y2-y1
	step = total_length/(num_points-1.0)
	middle_pentagon = num_points%2 == 0
	polygons = []
	if A == 180 and B == 270: # y2->y1 then x1->x2
		if not middle_pentagon: # odd
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points):
				polygons.append([(x1,y2-pi*step), (x1,y2-(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
			for pi in range(half_num_points):
				polygons.append([(x1+pi*step,y1), (x1+(pi+1)*step,y1), arc_pts[si+1], arc_pts[si]])
				si += 1
		if middle_pentagon: # even
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points-1):
				polygons.append([(x1,y2-pi*step), (x1,y2-(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
			pentagon = []
			pentagon.append((x1, y2-(half_num_points-1)*step))
			pentagon.append((x1, y1))
			pentagon.append((x1+step, y1))
			pentagon.append(arc_pts[si+1])
			pentagon.append(arc_pts[si])
			polygons.append(pentagon)
			si += 1
			for pi in range(1, half_num_points):
				polygons.append([(x1+pi*step,y1), (x1+(pi+1)*step,y1), arc_pts[si+1], arc_pts[si]])
				si += 1
	elif A == -90 and B == 0: # x1->x2 then y1->y2
		if not middle_pentagon: # odd
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points):
				polygons.append([(x1+pi*step,y1), (x1+(pi+1)*step,y1), arc_pts[si+1], arc_pts[si]])
				si += 1
			for pi in range(half_num_points):
				polygons.append([(x2,y1+pi*step), (x2,y1+(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
		if middle_pentagon: # even
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points-1):
				polygons.append([(x1+pi*step,y1), (x1+(pi+1)*step,y1), arc_pts[si+1], arc_pts[si]])
				si += 1
			pentagon = []
			pentagon.append((x1+(half_num_points-1)*step, y1))
			pentagon.append((x2, y1))
			pentagon.append((x2, y1+step))
			pentagon.append(arc_pts[si+1])
			pentagon.append(arc_pts[si])
			polygons.append(pentagon)
			si += 1
			for pi in range(1, half_num_points):
				polygons.append([(x2,y1+pi*step), (x2,y1+(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
	elif A == 0 and B == 90: # y1->y2 then x2->x1
		if not middle_pentagon: # odd
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points):
				polygons.append([(x2,y1+pi*step), (x2,y1+(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
			for pi in range(half_num_points):
				polygons.append([(x2-pi*step,y2), (x2-(pi+1)*step,y2), arc_pts[si+1], arc_pts[si]])
				si += 1
		if middle_pentagon: # even
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points-1):
				polygons.append([(x2,y1+pi*step), (x2,y1+(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
			pentagon = []
			pentagon.append((x2, y1+(half_num_points-1)*step))
			pentagon.append((x2, y2))
			pentagon.append((x2-step, y2))
			pentagon.append(arc_pts[si+1])
			pentagon.append(arc_pts[si])
			polygons.append(pentagon)
			si += 1
			for pi in range(1, half_num_points):
				polygons.append([(x2-pi*step,y2), (x2-(pi+1)*step,y2), arc_pts[si+1], arc_pts[si]])
				si += 1
	elif A == 90 and B == 180: # x2->x1 then y2->y1
		if not middle_pentagon: # odd
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points):
				polygons.append([(x2-pi*step,y2), (x2-(pi+1)*step,y2), arc_pts[si+1], arc_pts[si]])
				si += 1
			for pi in range(half_num_points):
				polygons.append([(x1,y2-pi*step), (x1,y2-(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
		if middle_pentagon: # even
			half_num_points = num_points//2
			si = 0
			for pi in range(half_num_points-1):
				polygons.append([(x2-pi*step,y2), (x2-(pi+1)*step,y2), arc_pts[si+1], arc_pts[si]])
				si += 1
			pentagon = []
			pentagon.append((x2-(half_num_points-1)*step))
			pentagon.append((x1, y2))
			pentagon.append((x1, y2-step))
			pentagon.append(arc_pts[si+1])
			pentagon.append(arc_pts[si])
			polygons.append(pentagon)
			si += 1
			for pi in range(1, half_num_points):
				polygons.append([(x1,y2-pi*step), (x1,y2-(pi+1)*step), arc_pts[si+1], arc_pts[si]])
				si += 1
	return polygons

# Given 5 values for a point (1 for itself and 4 for neighbors)
# construct polygons needed to draw this out; Lane width is lane_w
# and from this point draw only until distance d (half edge)
def construct_polygons(point_info, lane_w, d):
	neighbors = list(point_info[1:5])
	polygons = []
	if sum(neighbors) == 1:
		if neighbors == [False, False, False, True]:
			polygons += rect(-d, 0, 0, lane_w)
			polygons += rect(-d, 0, -lane_w, 0)
			polygons += outer_arc(0, 0, lane_w, -90, 90, num_points=5)
		elif neighbors == [False, False, True, False]:
			polygons += rect(-lane_w, 0, -d, 0)
			polygons += rect(0, lane_w, -d, 0)
			polygons += outer_arc(0, 0, lane_w, 0, 180, num_points=5)
		elif neighbors == [False, True, False, False]:
			polygons += rect(0, d, 0, lane_w)
			polygons += rect(0, d, -lane_w, 0)
			polygons += outer_arc(0, 0, lane_w, 90, 270, num_points=5)
		elif neighbors == [True, False, False, False]:
			polygons += rect(-lane_w, 0, 0, d)
			polygons += rect(0, lane_w, 0, d)
			polygons += outer_arc(0, 0, lane_w, -180, 0, num_points=5)
	elif sum(neighbors) == 2:
		if neighbors == [False, False, True, True]:
			polygons += rect(-d, 0, 0, lane_w)
			polygons += outer_arc(0, 0, lane_w, 0, 90, num_points=3)
			polygons += rect(0, lane_w, -d, 0)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
		elif neighbors == [False, True, False, True]:
			polygons += rect(-d, d, 0, lane_w)
			polygons += rect(-d, d, -lane_w, 0)
		elif neighbors == [True, True, False, False]:
			polygons += rect(-lane_w, 0, 0, d)
			polygons += outer_arc(0, 0, lane_w, 180, 270, num_points=3)
			polygons += rect(0, d, -lane_w, 0)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
		elif neighbors == [False, True, True, False]:
			polygons += rect(-lane_w, 0, -d, 0)
			polygons += outer_arc(0, 0, lane_w, 90, 180, num_points=3)
			polygons += rect(0, d, 0, lane_w)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
		elif neighbors == [True, False, True, False]:
			polygons += rect(-lane_w, 0, -d, d)
			polygons += rect(0, lane_w, -d, d)
		elif neighbors == [True, False, False, True]:
			polygons += rect(0, lane_w, 0, d)
			polygons += outer_arc(0, 0, lane_w, -90, 0, num_points=3)
			polygons += rect(-d, 0, -lane_w, 0)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
	elif sum(neighbors) == 3:
		if neighbors == [False, True, True, True]:
			polygons += rect(-d, d, 0, lane_w)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
		elif neighbors == [True, False, True, True]:
			polygons += rect(0, lane_w, -d, d)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
		elif neighbors == [True, True, False, True]:
			polygons += rect(-d, d, -lane_w, 0)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)			
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
		elif neighbors == [True, True, True, False]:
			polygons += rect(-lane_w, 0, -d, d)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)			
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
	elif sum(neighbors) == 4:
		if neighbors == [True, True, True, True]:
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
	return polygons

# https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
def make_counter_clockwise(polygon):
	esum = 0
	for pi in range(len(polygon)):
		x1, y1 = polygon[pi]
		x2, y2 = polygon[pi+1] if pi != len(polygon)-1 else polygon[0]
		esum += (x2-x1)*(y2+y1)
	return polygon if esum <= 0 else polygon[::-1]

def construct_grid(lattice, lane_w, edge_length, off_params):
	h, w = lattice.shape[0:2]
	h_off, w_off = off_params
	all_polygons = []
	for i in range(h):
		for j in range(w):
			if lattice[i][j][0]:
				curr_y, curr_x = i*edge_length, j*edge_length
				polygons = construct_polygons(lattice[i][j], lane_w, edge_length/2)
				polygons = [map(lambda pt: translate(pt, w_off+curr_x, h_off+curr_y), polygon) for polygon in polygons]
				polygons = map(make_counter_clockwise, polygons)
				polygons = [map(lambda pt: (round(pt[0], 2), round(pt[1], 2)), polygon) for polygon in polygons]
				# Inversion should not be needed
				all_polygons += polygons
	return all_polygons