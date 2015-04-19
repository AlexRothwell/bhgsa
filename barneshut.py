import numpy as np

class BHNTree(object):
    def __init__(self, theta, mids, side, dims):
        self.theta = theta
        self.desc += "-" + str(theta)
        self.mids = mids
        self.side = side
        self.body = None
        self.children = dict()
        self.dims = dims
        
    def insert(self, sol):
        
    def getForce(self, sol):
        force = np.zeros(self.dims)
        if self.isLeaf():
            if this.body is not None and this.body is not sol:
                force += addForce(this.body, sol)
            elif self.side / np.linalg.norm(sol.pos-self.body.pos) < self.theta:
                force += addForce(this.body, sol)
            else:
            {
                for (BHTree sub : subtrees.values())
                {
                    sub.updateForce(s,max);
                }
            }
    
    def isLeaf(self):
        return not self.children
    
    def addForce(self, sol_from, sol_to):
        dist = np.linalg.norm(sol_from.pos-sol_to.pos)
        return (sol_from.mass * (sol_from.pos - sol_to.pos) / dist)
        
    def getChildNum(self, sol):
        num = 0
        for i in range(dims):
            if sol.pos[i] < self.mids[i]:
                num <<= 1
            else:
                num = (num << 1) | 1
        return num


def bhForce(func, pop, kbest, G):
    side = getBounds(pop, func.dims, 1e-15)
    
    tree = BHNTree()
    for sol in pop:
        if sol in kbest:
            tree.insert(sol)
    
    for sol in pop:
        sol.force = tree.getForce(sol) * G
    Box box = new Box(mids,range);
    BHTree tree = new BHTree(box,dims,prob,theta);
    
    
def getBounds(pop, dims, eps):
    mids
    upper = np.array(pop[0].pos)
    lower = np.array(pop[0].pos)
    for sol in pop:
        for i in range(dims):
            if sol.pos[i] > upper[i]:
                upper[i] = sol.pos[i] + eps
            elif sol.pos[i] < lower[i]:
                lower[i] = sol.pos[i] - eps
    return lower, upper

if __name__ == '__main__':
    main()
else:
    bhForce.desc = "BH"