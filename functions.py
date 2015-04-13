from __future__ import division
from operator import mul
from math import floor, sin, cos, sqrt, exp, pi, e
import random

from solution import Solution


class Function(object):
    def __init__(self, num, upper, lower, dims, maximisation):
        self.num = num
        self.upper = upper
        self.lower = lower
        self.dims = dims
        self.max = maximisation
    
    def generateSolution(self):
        return Solution(self.dims, [random.uniform(self.lower[i],self.upper[i]) for i in range(self.dims)])
        
    def getFitnesses(self, pop):
        for sol in pop:
            sol.fit = self.getFitness(sol.pos)
            
    def inside(self, sol):
        for i in range(self.dims):
            if sol.pos[i] < self.lower[i] or sol.pos[i] > self.upper[i]:
                return False
        return True

def y(i,x):
    return 1 + (x[i] + 1)/4

def u(i, a, k, m):
    if i > a:
        return k * (i-a)**m
    elif i < -a:
        return k * (-i-a)**m
    else:
        return 0


def getFunction(func, n):
    if (func == 1):
        result =  Function(func, [100]*n, [-100]*n, n, False)
        result.getFitness = lambda x: sum([i*i for i in x])
        return result
    elif (func == 2):
        result = Function(func, [10]*n,[-10]*n, n, False)
        result.getFitness = lambda x: sum(map(abs,x)) + reduce(mul, map(abs,x))
        return result
    elif (func == 3):
        result = Function(func, [100]*n, [-100]*n, n, False)
        result.getFitness = lambda x: sum([sum([x[j] for j in range(0, i+1)])**2 for i in range(0,n)])
        return result
    elif (func == 4):
        result = Function(func, [100]*n, [-100]*n, n, False)
        result.getFitness = lambda x: max(map(abs,x))
        return result
    elif (func == 5):
        result = Function(func, [30]*n, [-30]*n, n, False)
        result.getFitness = lambda x: sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(0,n-1)])
        return result
    elif (func == 6):
        result = Function(func, [100]*n, [-100]*n, n, False)
        result.getFitness = lambda x: sum([floor(i + 0.5)**2 for i in x])
        return result
    elif (func == 7):
        result = Function(func, [1.28]*n, [-1.28]*n, n, False)
        result.getFitness = lambda x: sum([(i+1)*x[i]**4 + random.random() for i in range(0,n)])
        return result
    elif (func == 8):
        result = Function(func, [500]*n, [-500]*n, n, False)
        result.getFitness = lambda x: sum([-i*sin(sqrt(abs(i))) for i in x])
        return result
    elif (func == 9):
        result = Function(func, [5.12]*n, [-5.12]*n, n, False)
        result.getFitness = lambda x: sum([i*i - 10*cos(2*pi*i) + 10 for i in x])
        return result
    elif (func == 10):
        result = Function(func, [32]*n, [-32]*n, n, False)
        result.getFitness = lambda x: -20*exp(-0.2*sqrt(1/n*sum([i*i for i in x]))) - exp(1/n * sum([cos(2*pi*i) for i in x])) + 20 + e
        return result
    elif (func == 11):
        result = Function(func, [600]*n, [-600]*n, n, False)
        result.getFitness = lambda x: 1/4000*sum([i*i for i in x]) - reduce(mul, [cos(x[i]/(i+1)) for i in range(0,n)]) + 1
        return result
    elif (func == 12):
        result = Function(func, [50]*n, [-50]*n, n, False)
        result.getFitness = lambda x: pi/n * (10*sin(pi*y(0,x)) + sum([(y(i,x)-1)**2*(1+10*sin(pi*y(i,x))**2) for i in range(0,n)])) + \
            sum([u(i,10,100,4) for i in x])
        return result
    elif (func == 13):
        result = Function(func, [50]*n, [-50]*n, n, False)
        result.getFitness = lambda x: 0.1*(sin(3*pi*x[0])**2 + sum([(i-1)**2*(1+sin(3*pi*i+1)**2)for i in x]) + (x[n-1]-1)**2*(1+sin(2*pi*x[n-1]))) + sum([u(i,5,100,4) for i in x])
        return result
    elif (func == 14):
        result = Function(func, [65.53]*2, [-65.53]*2, 2, False)
        def fit(x):
            a = [[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],
                    [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]]
            res = 0
            for i in range(0,25):
                res += 1/(i+1 + (x[0]-a[0][i])**6 + (x[1]-a[1][i])**6)
            if res != 1/500:
                return 1/(1/500 + res)
            else:
                return 1/1e-15
        
        result.getFitness = fit
        return result
    elif (func == 15):
        result = Function(func, [5]*4, [-5]*4, 4, False)
        def fit(x):
            a=[.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
            b=[1/.25, 1/.5, 1/1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 16]
            res = 0
            for i in range(0,11):
                res += (a[i] - (x[0]*b[i]*(b[i]+x[1]))/(b[i]*b[i]+b[i]*x[2]+x[3]))**2
            return res
        result.getFitness = fit
        return result
    elif (func == 16):
        result = Function(func, [5]*2, [-5]*2, 2, False)
        result.getFitness = lambda x: 4*x[0]*x[0] + 2.1*x[0]**4 + 1/3*x[0]**6 + x[0]*x[1] - 4*x[1]*x[1] + 4*x[1]**4
        return result
    elif (func == 17):
        result = Function(func, [10,15], [-5,0], 2, False)
        result.getFitness = lambda x: (x[1]-(5.1/(4*pi*pi))*x[0]*x[0]+5/pi*x[0]-6)**2 + 10*(1-1/(8*pi))*cos(x[0]) + 10
        return result
    elif (func == 18):
        result = Function(func, [5]*2, [-5]*2, 2, False)
        result.getFitness = lambda x: (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]*x[0]-14*x[1]+6*x[0]*x[1]+3*x[1]*x[1]))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]*x[0]+48*x[1]-36*x[0]*x[1]+27*x[1]*x[1]))
        return result
    elif (func == 19):
        result = Function(func, [1]*3, [0]*3, 3, False)
        def fit(x):
            a=[[3, 10, 30],[.1, 10, 35],[3, 10, 30],[.1, 10, 35]]
            c=[1, 1.2, 3, 3.2]
            p=[[.3689, .117, .2673],[.4699, .4387, .747],[.1091, .8732, .5547],[.03815, .5743, .8828]]
            res = 0
            for i in range(0,4):
                res += c[i]*exp(-sum([a[i][j]*(x[j]-p[i][j])**2 for j in range(0,3)]))
            return -res
            
        result.getFitness = fit
        return result
    elif (func == 20):
        result = Function(func, [1]*6, [0]*6, 6, False)
        def fit(x):
            a=[[10, 3, 17, 3.5, 1.7, 8],[.05, 10, 17, .1, 8, 14],[3, 3.5, 1.7, 10, 17, 8],[17, 8, .05, 10, .1, 14]]
            c=[1, 1.2, 3, 3.2]
            p=[[.1312, .1696, .5569, .0124, .8283, .5886],[.2329, .4135, .8307, .3736, .1004, .9991],[.2348, .1415, .3522, .2883, .3047, .6650],[.4047, .8828, .8732, .5743, .1091, .0381]]
            res = 0
            for i in range(0,4):
                res += c[i]*exp(-sum([a[i][j]*(x[j]-p[i][j])**2 for j in range(0,6)]))
            return -res
            
        result.getFitness = fit
        return result
    elif (func == 21):
        result = Function(func, [10]*4, [0]*4, 4, False)
        def fit(x):
            a=[[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]]
            c=[.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
            res = 0
            for i in range(0,5):
                res += 1/(sum([x[j]-a[i][j] for j in range(0,4)]) + c[i])
            return -res

        result.getFitness = fit
        return result
    elif (func == 22):
        result = Function(func, [10]*4, [0]*4, 4, False)
        def fit(x):
            a=[[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]]
            c=[.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
            res = 0
            for i in range(0,7):
                res += 1/(sum([x[j]-a[i][j] for j in range(0,4)]) + c[i])
            return -res

        result.getFitness = fit
        return result
    elif (func == 23):
        result = Function(func, [10]*4, [0]*4, 4, False)
        def fit(x):
            a=[[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]]
            c=[.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
            res = 0
            for i in range(0,10):
                res += 1/(sum([x[j]-a[i][j] for j in range(0,4)]) + c[i])
            return -res

        result.getFitness = fit
        return result

def main():
    pass


if __name__ == '__main__':
    main()
