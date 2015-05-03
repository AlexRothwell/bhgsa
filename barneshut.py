import numpy as np
from solution import Solution

class BHNTree(object):
    def __init__(self, theta, mids, side, dims):
        self.theta = theta
        self.mids = mids
        self.side = side
        self.body = None
        self.children = dict()
        self.dims = dims
        
    def insert(self, sol):
        stack = [sol]
        trees = [self]
        
        while stack:
            curr = stack.pop()
            tree = trees.pop()
            children = tree.children
            #If there isn't a body already in this cell, set the body to curr
            if tree.body is None:
                tree.body = curr
            #If there is a body already in it and the node is not a leaf,
            # recursively add curr to the correct child node
            elif not tree.isLeaf():
                tree.body = getCentreMass(tree.body, curr)
                child_num = tree.getChildNum(curr)
                if child_num not in children:
                    children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                trees.append(children[child_num])
                stack.append(curr)
            #Otherwise, the node is an external node. So add both curr and the existing body
            # to the appropriate child nodes
            else:
                if tree.side / np.linalg.norm(tree.body.pos-curr.pos) > (1 << 2):
                    child_num = tree.getChildNum(tree.body)
                    children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                    trees.append(children[child_num])
                    stack.append(tree.body)
                    child_num = tree.getChildNum(curr)
                    if child_num not in children:
                        children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                    trees.append(children[child_num])
                    stack.append(curr)
                tree.body = getCentreMass(curr, tree.body)
        
    def getForce(self, sol):
        force = np.zeros(self.dims)
        if self.isLeaf():
            if self.body is not None and self.body is not sol:
                force += addForce(self.body, sol)
        elif self.side / np.linalg.norm(sol.pos-self.body.pos) < self.theta:
            force += addForce(self.body, sol)
        else:
            for child in self.children.values():
                force += child.getForce(sol)
        return force
    
    def isLeaf(self):
        return not self.children
        
    def getChildNum(self, sol):
        num = 0
        for i in range(self.dims):
            if sol.pos[i] < self.mids[i]:
                num <<= 1
            else:
                num = (num << 1) | 1
        return num

class BHForce(object):
    def __init__(self, theta):
        self.theta = theta
        self.desc += "-" + str(theta)
    
    def __call__(self, func, pop, kbest, G):
        lower, upper = getBounds(kbest, func.dims, 1e-15)
        range = upper - lower
        mids = np.array([upper - range/2]*func.dims)
        
        tree = BHNTree(self.theta, mids, range, func.dims)
        for sol in kbest:
            tree.insert(sol)
        
        for sol in pop:
            sol.force = tree.getForce(sol) * G

    
def getBounds(pop, dims, eps):
    upper = 0
    lower = 0
    for sol in pop:
        for i in range(dims):
            if sol.pos[i] > upper:
                upper = sol.pos[i] + eps
            elif sol.pos[i] < lower:
                lower = sol.pos[i] - eps
    return lower, upper
    
def addForce(sol_from, sol_to):
    dist = np.linalg.norm(sol_from.pos-sol_to.pos)
    return (sol_from.mass * (sol_from.pos - sol_to.pos) / dist)
        
def getMids(tree, num):
    mids = np.copy(tree.mids)
    for dim in range(tree.dims):
        mids[dim] -= tree.side/4 + getBit(num,dim)*tree.side/2
    return mids
        
def getBit(num, i):
    return (num >> i) & 1

def getCentreMass(body1, body2):
    comb_mass = body1.mass + body2.mass
    centre_pos = (body1.pos * body1.mass + body2.pos * body2.mass)/comb_mass
    return Solution(body1.dims, centre_pos, comb_mass)

if __name__ == '__main__':
    main()
else:
    BHForce.desc = "BH"