from Box2D.b2 import contactListener
import numpy as np

ROAD_COLOR = [0.4, 0.4, 0.4]

# Class to specify what to do when we encounter
# a static body
class FrictionDetector(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile: return
        # show a streak behind the object with color ROAD_COLOR
        tile.color = ROAD_COLOR[:]
        if not obj or "tiles" not in obj.__dict__: return

        # Assign last tile direction
        epsilon = 0.01
        if tile != None and "direction" in tile.__dict__:
            for car_idx in range(len(self.env.cars)):
                if self.env.cars[car_idx] == None or self.env.cars[car_idx].hull == None:
                    continue
                vx, vy = obj.linearVelocity
                cvx, cvy = self.env.cars[car_idx].hull.linearVelocity
                if np.abs(vx-cvx) <= epsilon and np.abs(vy-cvy) <= epsilon:
                    self.env.last_tile_direction[car_idx] = tile.direction

        # Should not be needed since lane separators are
        # polylines on the world.
        ## if tile.boxtype == "lane-sep": return
        
        if begin:
            obj.tiles.add(tile)
            
            # TODO: change how this reward is computed
            self.env.reward += 1000.0/len(self.env.track)
        else:
            obj.tiles.remove(tile)
