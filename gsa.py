import random
from timeit import default_timer
import numpy as np
import pandas as pd
from math import exp, ceil, sqrt

from solution import Solution

G_0 = 100
ALPHA = 20
MAX_T = 1000
FIXED_TIME = 5
OUTPUT_RATE = 100
OUTPUT_FILE = "out-"


def update(func, pop, t):
    best, worst, G = 0, 0, 0
    if func.max:
        best = max(sol.fit for sol in pop)
        worst = min(sol.fit for sol in pop)
    else:
        best = min(sol.fit for sol in pop)
        worst = max(sol.fit for sol in pop)
    G = G_0 * exp(-ALPHA * t/MAX_T)
    return best, worst, G

def updateBest(func, pop, t):
    #Ensure that k is always equal to or greater than 1 and smaller than population size
    k = int(ceil(len(pop) * (1 - t/MAX_T)))
    if (func.max):
        pop.sort(key=lambda sol: sol.fit, reverse=True)
    else:
        pop.sort(key=lambda sol: sol.fit)
    return pop[0:k]

def calcMasses(pop, best, worst):
    div = best - worst
    threshold = 1e-15
    s = 0
    if abs(div) < threshold:
        for sol in pop:
            sol.mass = 1
            s += 1
    else:
        for sol in pop:
            mass = abs(sol.fit - worst)/div
            sol.mass = mass
            s += mass
    for sol in pop:
        sol.mass /= s

def calcMovement(func, pop):
    for sol in pop:
        sol.vel *= np.random.random(func.dims)
        sol.vel += sol.force
        sol.pos += sol.vel

def reinitialiseSolutions(func, pop):
    for i in range(len(pop)):
        if not func.inside(pop[i]):
            pop[i] = func.generateSolution()

def initialise(func, size):
    return [func.generateSolution() for i in range(size)]
    
def formatProgress(t, pop, columns):
    data = list(map(str,[t] + [sol.fit for sol in pop]))
    return pd.Series(dict(zip(columns,data)))
    
def output(best, t, force, func, conditions):
    data = list(map(str,[func.num, force.desc, best, t]))
    return pd.Series(dict(zip(["FUNCTION","SOLVER","BEST","ITERATIONS"],data)))

def getDist(sol, kth):
    return sqrt(np.sum((sol-kth)**2))
    
def basicForce(func, pop, kbest, G):
    for sol in pop:
        pos = sol.pos
        sol.force = np.zeros(func.dims)
        for kth in kbest:
            if sol is not kth:
                kth_pos = kth.pos
                dist = getDist(pos,kth_pos)
                sol.force += np.random.random(func.dims) * G * kth.mass * (kth_pos - pos) / dist
    
def iterationCondition(step, time):
    return step < MAX_T
    
def timeCondition(step, time):
    return time < FIXED_TIME
    
#takes the function, population size, force calculator, stop condition, seed
def gsa(function, pop_size, force, conditions, seed = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    population = initialise(function, pop_size)
    function.getFitnesses(population)

    t = 0
    #Track progress
    progColumns = ["t"] + ["fitness" + str(i) for i in range(1,pop_size + 1)]
    progress = pd.DataFrame(columns = progColumns)
    progress = progress.append(formatProgress(t, population, progColumns), ignore_index = True)
    start = default_timer()
    while (conditions(t,default_timer() - start)):
        #Update functions
        best, worst, G = update(function, population, t)

        #Work out kbest
        kbest = updateBest(function, population, t)
        
        #Mass calculations
        calcMasses(population, best, worst)

        #Force calculations
        force(function, population, kbest, G)

        #Calculate velocity and position
        calcMovement(function, population)

        #Remove solutions which are out of bounds
        reinitialiseSolutions(function, population)

        t += 1
        
        #Evaluate fitnesses
        function.getFitnesses(population)
        
        if (t % OUTPUT_RATE == 0):
            #output current data
            progress = progress.append(formatProgress(t, population, progColumns), ignore_index = True)
            
    progress.to_csv(function.desc + ".csv", index = False)
    return output(best, t, force, function, conditions)
    


def main():
    pass


if __name__ == '__main__':
    main()
else:
    iterationCondition.desc = "fixed-it"
    timeCondition.desc = "fixed-time"
    basicForce.desc = "GSA"
