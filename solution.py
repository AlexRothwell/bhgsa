#class Solution(object):
#    def __init__(self, dims, pos):
#        self.pos = np.array(pos, copy=True)
#        self.force = np.array([None] * dims)
#        self.vel = np.array(self.force, copy=True)

#    def __init__(self, dims, pos, mass):
#        self.pos = np.array(pos, copy=True)
#        self.force = np.array([None] * dims)
#        self.vel = np.array(self.force, copy=True)
#        self.mass = mass

#    def __repr__(self):
#        return ",".join(self.pos.tolist())

class Solution(object):
    def __init__(self,dims,pos, mass = None):
        self.pos = pos
        self.force = [0]*dims
        self.vel = [0]*dims
        self.mass = mass

    def __repr__(self):
        return ",".join(self.pos)