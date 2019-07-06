import sys, math
import numpy as np
import random
from collections import deque
from copy import deepcopy

# BFS to check graph connectivity
def BFS(lattice, edges=None):
	h, w = lattice.shape
	tovisit = deque()
	for i in range(h):
		for j in range(w):
			if lattice[i][j]:
				tovisit.append((i, j))
				break
		if len(tovisit) == 1: break
	if len(tovisit) == 0: return False # empty graph
	n = np.sum(lattice)
	visited = set()
	while len(tovisit) > 0:
		curr = tovisit.popleft()
		visited.add(curr)
		i, j = curr
		if i-1 >= 0 and (i-1, j) not in visited and lattice[i-1, j]:
			if not edges or (not (edges and ((i-1, j, i, j, 1) in edges or (i, j, i-1, j, 3) in edges))):
				tovisit.append((i-1, j))
		if i+1 < h and (i+1, j) not in visited and lattice[i+1, j]:
			if not edges or (not (edges and ((i+1, j, i, j, 3) in edges or (i, j, i+1, j, 1) in edges))):
				tovisit.append((i+1, j))
		if j-1 >= 0 and (i, j-1) not in visited and lattice[i, j-1]:
			if not edges or (not (edges and ((i, j-1, i, j, 2) in edges or (i, j, i, j-1, 4) in edges))):
				tovisit.append((i, j-1))
		if j+1 < w and (i, j+1) not in visited and lattice[i, j+1]:
			if not edges or (not (edges and ((i, j+1, i, j, 4) in edges or (i, j, i, j+1, 2) in edges))):
				tovisit.append((i, j+1))
	return len(visited) == n

# Lattice is a set of h x w points, randomly set or unset. We also 
# encode neighbor information. Overall the dimensionality is
# h x w x 5, where (i, j, 0) denotes value of point (i, j). while
# (i, j, 1:4) denote values of neighbors going top, right, bottom, left.
#
# Lattice constraints have min max values for 5 types of junctions.
#
# After constructing the lattice, do random deletions until those conditions are met.
def construct_lattice(h, w, p, lattice_constraints, num_deletions, pre_provided_lattice=None, delete_edges=None):

	if pre_provided_lattice != None:
		pre_provided_lattice = np.array(pre_provided_lattice, dtype=bool)
		assert(len(pre_provided_lattice.shape) == 2)
		h, w = pre_provided_lattice.shape

	lattice = np.zeros(shape=(h, w, 5), dtype=bool)
	counts_loop = 0
	while True:
		counts_loop += 1

		for i in range(h):
			for j in range(w):
				if pre_provided_lattice is not None:
					lattice[i][j][0] = pre_provided_lattice[i][j]
				else:
					lattice[i][j][0] = bool(np.random.choice(2, 1, p=[1-p, p])[0])

		is_connected = BFS(lattice[:, :, 0])
		if not is_connected:
			if pre_provided_lattice is not None:
				print("Provided lattice isn't valid!")
				exit(0)
			continue

		edges = []
		for i in range(h):
			for j in range(w):
				# update bottom neighbor
				if i-1 >= 0:
					lattice[i-1][j][1] = lattice[i][j][0]
					if (i, j, i-1, j, 3) not in edges: edges.append((i-1, j, i, j, 1))
				# update right neighbor
				if j+1 < w:
					lattice[i][j+1][4] = lattice[i][j][0]
					if (i, j, i, j+1, 2) not in edges: edges.append((i, j+1, i, j, 4))
				# update top neighbor
				if i+1 < h:
					lattice[i+1][j][3] = lattice[i][j][0]
					if (i, j, i+1, j, 1) not in edges: edges.append((i+1, j, i, j, 3))
				# update left neighbor
				if j-1 >= 0:
					lattice[i][j-1][2] = lattice[i][j][0]
					if (i, j, i, j-1, 4) not in edges: edges.append((i, j-1, i, j, 2))

		if pre_provided_lattice is not None:
			if delete_edges is not None:
				for edge in delete_edges:
					a, b, c, d = edge
					assert(0 <= a < h and 0 <= b < w and 0 <= c < h and 0 <= d < w, "invalid edge: %s" % str(edge))
					if a+1 == c: ntype, otype = 1, 3
					if b-1 == d: ntype, otype = 4, 2
					if a-1 == c: ntype, otype = 3, 1
					if b+1 == d: ntype, otype = 2, 4
					lattice[a][b][ntype] = False
					lattice[c][d][otype] = False
			break

		# check constraints before random deletions
		is_satisfied = check_constraints(lattice, lattice_constraints)
		if not is_satisfied: continue

		# random deletions; it is possible to loop forever if the graph cannot satisfy lattice
		# constraints ever, after deleting the required number of edges
		lattice, success = do_random_deletions(lattice, np.array(edges), num_deletions, lattice_constraints)
		if not success: continue
		else: break

	if pre_provided_lattice is None: print('Took %d iterations to generate track' % counts_loop)
	return lattice

# Do random deletions, given number of deletions
# Ensure that connectedness holds
# Deletions are done by choosing a random subset of the edges and deleting those
def do_random_deletions(lattice, edges_orig, n, constraints, max_tries=1000):
	num_tries = 0
	while num_tries < max_tries:
		new_lattice = deepcopy(lattice)
		edges = deepcopy(edges_orig)
		edges_to_delete = edges[np.random.choice(len(edges), n, replace=False)].tolist()
		for (a, b, c, d, _) in edges_to_delete:
			if a+1 == c: ntype, otype = 1, 3
			if b-1 == d: ntype, otype = 4, 2
			if a-1 == c: ntype, otype = 3, 1
			if b+1 == d: ntype, otype = 2, 4
			new_lattice[a][b][ntype] = False
			new_lattice[c][d][otype] = False
		new_edges = []
		edges = edges.tolist()
		for edge in edges:
			if edge not in edges_to_delete:
				new_edges.append(edge)
		edges = new_edges
		num_tries += 1
		is_connected = BFS(new_lattice[:, :, 0], edges=edges)
		if not is_connected: continue
		is_satisfied = check_constraints(new_lattice, constraints)
		if not is_satisfied: continue
		else: 
			# print('Took %d inner iterations for random deletions' % num_tries)
			return new_lattice, True
	# print('Took %d inner iterations for random deletions' % num_tries)
	return lattice, False

# Check constraints on the degree of nodes in a lattice
def check_constraints(lattice, lattice_constraints):
	h, w = lattice.shape[:2]
	# count neighbors
	nc = []
	for i in range(h):
		for j in range(w):
			if lattice[i, j, 0]:
				nc.append(sum(lattice[i, j, 1:]))
	satisfied = [lattice_constraints[i][0] <= nc.count(i) <= lattice_constraints[i][1] for i in range(5)]
	is_satisfied = sum(satisfied) == 5
	return is_satisfied

# Create lanes for each edge from this point, leaving 2*lane_w
# on each side; d is the half edge
def generate_lane_sep(pt_info, lane_w, d, lane_sep):
	lane_polygons = []
	if pt_info[0]:
		if pt_info[1]:
			lane_polygons += rect(-lane_sep, lane_sep, 2*lane_w, d, 2, 2)
		if pt_info[2]:
			lane_polygons += rect(2*lane_w, d, -lane_sep, lane_sep, 2, 2)
		if pt_info[3]:
			lane_polygons += rect(-lane_sep, lane_sep, -d, -2*lane_w, 2, 2)
		if pt_info[4]:
			lane_polygons += rect(-d, -2*lane_w, -lane_sep, lane_sep, 2, 2)
	return lane_polygons

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

# translate to some place after rotating by angle degrees
def translate(pt, xs, ys, angle=0):
	pt = (
		xs+math.cos(angle*math.pi/180.0)*pt[0]-math.sin(angle*math.pi/180.0)*pt[1],
		ys+math.sin(angle*math.pi/180.0)*pt[0]+math.cos(angle*math.pi/180.0)*pt[1]
	)
	return pt

def extended_rect(x1, x2, y1, y2, xpoints=2, ypoints=2):
	if x1 == 0 and x2 > 2*lane_w:
		return rect(x1, 2*lane_w, y1, y2, xpoints=xpoints, ypoints=ypoints)+\
			rect(2*lane_w, x2, y1, y2, xpoints=xpoints, ypoints=ypoints)
	if x1 < -2*lane_w and x2 == 0:
		return rect(x1, -2*lane_w, y1, y2, xpoints=xpoints, ypoints=ypoints)+\
			rect(-2*lane_w, x2, y1, y2, xpoints=xpoints, ypoints=ypoints)
	if y1 == 0 and y2 > 2*lane_w:
		return rect(x1, x2, y1, 2*lane_w, xpoints=xpoints, ypoints=ypoints)+\
			rect(x1, x2, 2*lane_w, y2, xpoints=xpoints, ypoints=ypoints)
	if y1 < -2*lane_w and y2 == 0:
		return rect(x1, x2, y1, -2*lane_w, xpoints=xpoints, ypoints=ypoints)+\
			rect(x1, x2, -2*lane_w, y2, xpoints=xpoints, ypoints=ypoints)

def rect(x1, x2, y1, y2, xpoints=2, ypoints=2):
	assert(x1 <= x2)
	assert(y1 <= y2)
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
	special_rectangles = [] # these are basically parts of junctions
	# For each rectangle provide which direction is considered as
	# going right
	if sum(neighbors) == 1:
		if neighbors == [False, False, False, True]:
			polygons += rect(-d, 0, 0, lane_w)
			polygons += rect(-d, 0, -lane_w, 0)
			polygons += outer_arc(0, 0, lane_w, -90, 90, num_points=5)
			direction = "lrnnnn"
		elif neighbors == [False, False, True, False]:
			polygons += rect(-lane_w, 0, -d, 0)
			polygons += rect(0, lane_w, -d, 0)
			polygons += outer_arc(0, 0, lane_w, 0, 180, num_points=5)
			direction = "btnnnn"
		elif neighbors == [False, True, False, False]:
			polygons += rect(0, d, 0, lane_w)
			polygons += rect(0, d, -lane_w, 0)
			polygons += outer_arc(0, 0, lane_w, 90, 270, num_points=5)
			direction = "lrnnnn"
		elif neighbors == [True, False, False, False]:
			polygons += rect(-lane_w, 0, 0, d)
			polygons += rect(0, lane_w, 0, d)
			polygons += outer_arc(0, 0, lane_w, -180, 0, num_points=5)
			direction = "btnnnn"
	elif sum(neighbors) == 2:
		if neighbors == [False, False, True, True]:
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			special_rectangles += rect(-2*lane_w, 0, 0, lane_w)
			polygons += outer_arc(0, 0, lane_w, 0, 90, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
			special_rectangles += rect(0, lane_w, -2*lane_w, 0)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			direction = "lnntbnnr"
		elif neighbors == [False, True, False, True]:
			polygons += rect(-d, d, 0, lane_w)
			polygons += rect(-d, d, -lane_w, 0)
			direction = "lr"
		elif neighbors == [True, True, False, False]:
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			special_rectangles += rect(-lane_w, 0, 0, 2*lane_w)
			polygons += outer_arc(0, 0, lane_w, 180, 270, num_points=3)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			special_rectangles += rect(0, 2*lane_w, -lane_w, 0)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
			direction = "bnnrlnnt"
		elif neighbors == [False, True, True, False]:
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			special_rectangles += rect(-lane_w, 0, -2*lane_w, 0)
			polygons += outer_arc(0, 0, lane_w, 90, 180, num_points=3)
			polygons += rect(2*lane_w, d, 0, lane_w)
			special_rectangles += rect(0, 2*lane_w, 0, lane_w)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
			direction = "bnnlrnnt"
		elif neighbors == [True, False, True, False]:
			polygons += rect(-lane_w, 0, -d, d)
			polygons += rect(0, lane_w, -d, d)
			direction = "bt"
		elif neighbors == [True, False, False, True]:
			polygons += rect(0, lane_w, 2*lane_w, d)
			special_rectangles += rect(0, lane_w, 0, 2*lane_w)
			polygons += outer_arc(0, 0, lane_w, -90, 0, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			special_rectangles += rect(-2*lane_w, 0, -lane_w, 0)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			direction = "tnnrlnnb"
	elif sum(neighbors) == 3:
		if neighbors == [False, True, True, True]:
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += rect(2*lane_w, d, 0, lane_w)
			special_rectangles += rect(-2*lane_w, 0, 0, lane_w)
			special_rectangles += rect(0, 2*lane_w, 0, lane_w)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
			direction = "llbnnrrnnt"
		elif neighbors == [True, False, True, True]:
			polygons += rect(0, lane_w, -d, -2*lane_w)
			polygons += rect(0, lane_w, 2*lane_w, d)
			special_rectangles += rect(0, lane_w, -2*lane_w, 0)
			special_rectangles += rect(0, lane_w, 0, 2*lane_w)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += inner_arc(-2*lane_w, 0, -2*lane_w, 0, -2*lane_w, -2*lane_w, lane_w, 0, 90, num_points=3)
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			direction = "ttlnnbbnnr"
		elif neighbors == [True, True, False, True]:
			polygons += rect(-d, -2*lane_w, -lane_w, 0)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			special_rectangles += rect(-2*lane_w, 0, -lane_w, 0)
			special_rectangles += rect(0, 2*lane_w, -lane_w, 0)
			polygons += rect(-d, -2*lane_w, 0, lane_w)
			polygons += inner_arc(-2*lane_w, 0, 0, 2*lane_w, -2*lane_w, 2*lane_w, lane_w, -90, 0, num_points=3)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
			direction = "rrlnnbrnnt"
		elif neighbors == [True, True, True, False]:
			polygons += rect(-lane_w, 0, -d, -2*lane_w)
			polygons += rect(-lane_w, 0, 2*lane_w, d)
			special_rectangles += rect(-lane_w, 0, -2*lane_w, 0)
			special_rectangles += rect(-lane_w, 0, 0, 2*lane_w)
			polygons += rect(2*lane_w, d, 0, lane_w)
			polygons += inner_arc(0, 2*lane_w, 0, 2*lane_w, 2*lane_w, 2*lane_w, lane_w, 180, 270, num_points=3)
			polygons += rect(0, lane_w, 2*lane_w, d)
			polygons += rect(2*lane_w, d, -lane_w, 0)
			polygons += inner_arc(0, 2*lane_w, -2*lane_w, 0, 2*lane_w, -2*lane_w, lane_w, 90, 180, num_points=3)
			polygons += rect(0, lane_w, -d, -2*lane_w)
			direction = "bblnntrnnt"
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
			direction = "bnnrlnntrnntlnnb"
	return polygons, direction, special_rectangles

# https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
def make_counter_clockwise(polygon):
	esum = 0
	for pi in range(len(polygon)):
		x1, y1 = polygon[pi]
		x2, y2 = polygon[pi+1] if pi != len(polygon)-1 else polygon[0]
		esum += (x2-x1)*(y2+y1)
	return polygon if esum <= 0 else polygon[::-1]

def construct_grid(lattice, lane_w, edge_length, off_params, lane_sep):
	h, w = lattice.shape[0:2]
	h_off, w_off = off_params
	normal_polygons = []
	special_polygons = []
	lane_sep_polygons = []
	directions = ""
	relevant_nodes = []
	special_relevant_nodes = []
	for i in range(h):
		for j in range(w):
			if lattice[i][j][0]:
				curr_y, curr_x = i*edge_length, j*edge_length
				polygons, direction, special_rectangles = construct_polygons(lattice[i, j], lane_w, edge_length/2)
				directions += direction
				relevant_nodes += [(i, j),]*len(direction)
				special_relevant_nodes += [(i, j),]*len(special_rectangles)
				# Polygons
				polygons = [list(map(lambda pt: translate(pt, w_off+curr_x, h_off+curr_y), polygon)) for polygon in polygons]
				polygons = list(map(make_counter_clockwise, polygons))
				polygons = [list(map(lambda pt: (round(pt[0], 2), round(pt[1], 2)), polygon)) for polygon in polygons]
				# Special Rectangles
				special_rectangles = [list(map(lambda pt: translate(pt, w_off+curr_x, h_off+curr_y), polygon)) for polygon in special_rectangles]
				special_rectangles = list(map(make_counter_clockwise, special_rectangles))
				special_rectangles = [list(map(lambda pt: (round(pt[0], 2), round(pt[1], 2)), polygon)) for polygon in special_rectangles]
				# Lane separator polygons
				ls_polygons = generate_lane_sep(lattice[i, j], lane_w, edge_length/2, lane_sep)
				ls_polygons = [list(map(lambda pt: translate(pt, w_off+curr_x, h_off+curr_y), polygon)) for polygon in ls_polygons]
				ls_polygons = list(map(make_counter_clockwise, ls_polygons))
				ls_polygons = [list(map(lambda pt: (round(pt[0], 2), round(pt[1], 2)), polygon)) for polygon in ls_polygons]
				# Inversion should not be needed
				normal_polygons += polygons
				special_polygons += special_rectangles
				lane_sep_polygons += ls_polygons
	return normal_polygons, special_polygons, lane_sep_polygons, directions, relevant_nodes, special_relevant_nodes

# shape is a circle and a triangle within
# 4 states: straight, left, right, stop
class TrafficLight:
	# pos = how much to translate
	# state cycle defines transition, happens every limit number of steps
	# also needs initial pos, rotation
	# r is the outer circle radius, r2 is the max dimension of inner thing
	def __init__(self, state_cycle, r, r2, pos, rot=0, limit=100):
		self.state_cycle = state_cycle
		self.state_idx = 0
		self.limit = limit
		self.idx = 0
		self.circle = arc_points(0, 330, 0, 0, r, num_points=12)
		half_side = int(r2/(math.sqrt(2)))
		self.stop = rect(-half_side, half_side, -half_side, half_side, 2, 2)[0]
		self.right = arc_points(0, 240, 0, 0, r2, num_points=3)
		self.right += [self.right[-1]]
		self.straight = arc_points(90, 330, 0, 0, r2, num_points=3)
		self.straight += [self.straight[-1]]
		self.left = arc_points(60, 300, 0, 0, r2, num_points=3)
		self.left += [self.left[-1]]
		self.pos = pos
		self.rot = rot

	def get_polygons(self, shift_pos):
		self.idx = (self.idx+1)%self.limit
		if self.idx == 0:
			self.state_idx = (self.state_idx+1)%len(self.state_cycle)
		curr_state = self.state_cycle[self.state_idx]
		to_return = [self.circle, eval('self.%s' % curr_state)]
		to_return = [[translate(pt, *self.pos, angle=self.rot) for pt in polygon] for polygon in to_return]
		to_return = [[translate(pt, *shift_pos) for pt in polygon] for polygon in to_return]
		to_return = list(map(make_counter_clockwise, to_return))
		return to_return

	def shifted_pos(self, shift_pos):
		return translate(self.pos, *shift_pos)

# given neighbor information, construct traffic light objects around (0, 0)
def construct_traffic_lights(neighbors, lane_w, r, r2):
	lights = []
	shorthand = {'s': "straight", 'l': "left", 'r': "right", 'n': "stop"}
	create_lights_cycle = lambda s: list(map(lambda k: shorthand[k], list(s)))
	if list(neighbors) == [True, True, True, True]:
		lights.append(TrafficLight(create_lights_cycle("srlnnnnn"), r, r2, (lane_w*2.5, lane_w*0.5), 90))
		lights.append(TrafficLight(create_lights_cycle("nnnnsrnl"), r, r2, (lane_w*0.5, lane_w*-2.5), 0))
		lights.append(TrafficLight(create_lights_cycle("srnlnnnn"), r, r2, (lane_w*-2.5, lane_w*-0.5), -90))
		lights.append(TrafficLight(create_lights_cycle("nnnnsrln"), r, r2, (lane_w*-0.5, lane_w*2.5), 180))
	elif list(neighbors) == [False, True, True, True]:
		lights.append(TrafficLight(create_lights_cycle("ssnl"), r, r2, (lane_w*2.5, lane_w*0.5), 90))
		lights.append(TrafficLight(create_lights_cycle("nrlr"), r, r2, (lane_w*0.5, lane_w*-2.5), 0))
		lights.append(TrafficLight(create_lights_cycle("srrn"), r, r2, (lane_w*-2.5, lane_w*-0.5), -90))
	elif list(neighbors) == [True, False, True, True]:
		lights.append(TrafficLight(create_lights_cycle("ssnl"), r, r2, (lane_w*0.5, lane_w*-2.5), 0))
		lights.append(TrafficLight(create_lights_cycle("nrlr"), r, r2, (lane_w*-2.5, lane_w*-0.5), -90))
		lights.append(TrafficLight(create_lights_cycle("srrn"), r, r2, (lane_w*-0.5, lane_w*2.5), 180))
	elif list(neighbors) == [True, True, False, True]:
		lights.append(TrafficLight(create_lights_cycle("ssnl"), r, r2, (lane_w*-2.5, lane_w*-0.5), -90))
		lights.append(TrafficLight(create_lights_cycle("nrlr"), r, r2, (lane_w*-0.5, lane_w*2.5), 180))
		lights.append(TrafficLight(create_lights_cycle("srrn"), r, r2, (lane_w*2.5, lane_w*0.5), 90))
	elif list(neighbors) == [True, True, True, False]:
		lights.append(TrafficLight(create_lights_cycle("ssnl"), r, r2, (lane_w*-0.5, lane_w*2.5), 180))
		lights.append(TrafficLight(create_lights_cycle("nrlr"), r, r2, (lane_w*2.5, lane_w*0.5), 90))
		lights.append(TrafficLight(create_lights_cycle("srrn"), r, r2, (lane_w*0.5, lane_w*-2.5), 0))
	return lights

