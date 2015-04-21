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
        stack = [sol]
        trees = [self]
        
        while stack:
            curr = stack.pop()
            tree = trees.pop()
            children = tree.children
            if tree.body is None:
                tree.body = curr
            elif not tree.isLeaf():
                tree.body = getCentreMass(tree.body, curr)
                child_num = tree.getChildNum(curr)
                if child_num not in children:
                    children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                trees.push(children[child_num])
                stack.push(curr)
            else:
                if tree.side / np.linalg.norm(tree.body.pos-curr.pos) < 1 << 5:
                    child_num = tree.getChildNum(tree.body)
                    children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                    trees.push(children[child_num])
                    stack.push(tree.body)
                    child_num = tree.getChildNum(curr)
                    if child_num not in children:
                        children[child_num] = BHNTree(tree.theta, getMids(tree, child_num), tree.side/2, tree.dims)
                    trees.push(children[child_num])
                    stack.push(curr)
                tree.body = getCentreMass(curr, tree.body)
        
    def getForce(self, sol):
        force = np.zeros(self.dims)
        if self.isLeaf():
            if this.body is not None and this.body is not sol:
                force += addForce(this.body, sol)
        elif self.side / np.linalg.norm(sol.pos-self.body.pos) < self.theta:
            force += addForce(this.body, sol)
        else:
            for child in self.children.values():
                force += child.getForce(sol)
    
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
    lower, upper = getBounds(kbest, func.dims, 1e-15)
    range = upper - lower
    mids = np.array([upper - range/2 for i in range(func.dims)])
    
    tree = BHNTree(theta, mids, range, func.dims)
    if sol in kbest:
        tree.insert(sol)
    
    for sol in pop:
        sol.force = tree.getForce(sol) * G
    
    Box box = new Box(mids,range);
    BHTree tree = new BHTree(box,dims,prob,theta);
    
    
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
    
def getMids(tree, num):

def getCentreMass(body1, body2):
    

if __name__ == '__main__':
    main()
else:
    bhForce.desc = "BH"