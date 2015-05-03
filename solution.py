import numpy as np

class Solution(object):
    def __init__(self, dims, pos, mass = None):
        self.pos = np.array(pos, copy=True, dtype=float)
        self.force = np.array([0] * dims, dtype=float)
        self.vel = np.array(self.force, copy=True, dtype=float)
        self.mass = mass
        self.dims = dims

    def __repr__(self):
        return ",".join(self.pos.tolist())