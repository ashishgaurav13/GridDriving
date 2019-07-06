from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Determine whether (x, y) can be localized to
# any polygon in road_poly
def on_road(road_poly, x, y):
	our_pt = Point(x, y)
	for pid, polygon in enumerate(road_poly):
		if Polygon(polygon).contains(our_pt):
			return True, pid
	return False, -1

# Are we on road or off road? mostly convenience
def determine_road(lattice, edge_length, road_poly, pos):

	# x, y from pos
	x, y = pos

	# Make sure (x, y) is localizable
	new_road_poly = list(map(lambda el: el[0], road_poly))
	return on_road(new_road_poly, x, y)