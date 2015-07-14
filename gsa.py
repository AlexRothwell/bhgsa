import random
from timeit import default_timer
import numpy as np
import pandas as pd
from math import exp, ceil, sqrt

from solution import Solution

G_0 = 100
ALPHA = 20
MAX_T = 1000
FIXED_TIME_SHORT = 5
FIXED_TIME_LONG = 20
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
    
def formatProgress(t, time, pop, columns):
    data = list(map(str,[t,time] + [sol.fit for sol in pop]))
    return pd.Series(dict(zip(columns,data)))
    
def output(columns, best, t, force, func, conditions, time):
    data = list(map(str,[func.num, force.desc, best, t, time]))
    return pd.Series(dict(zip(columns,data)))

def getDist(sol, kth):
    return sqrt(np.sum((sol-kth)**2))
    
def calculateBasicForce(dims, sol, kbest, G):
    pos = sol.pos
    result = np.zeros(dims)
    for kth in kbest:
        if sol is not kth:
            kth_pos = kth.pos
            dist = getDist(pos,kth_pos)
            result += np.random.random(dims) * G * kth.mass * (kth_pos - pos) / dist
    return result
    
def basicForce(func, pop, kbest, G):
    for sol in pop:
        sol.force = calculateBasicForce(func.dims, sol, kbest, G)
    
def iterationCondition(step, time):
    return step < MAX_T
    
def iterationUpdate(func, pop, step, time):
    best, worst, G = 0, 0, 0
    if func.max:
        best = max(sol.fit for sol in pop)
        worst = min(sol.fit for sol in pop)
    else:
        best = min(sol.fit for sol in pop)
        worst = max(sol.fit for sol in pop)
    G = G_0 * exp(-ALPHA * step/MAX_T)
    k = int(ceil(len(pop) * (1 - step/MAX_T)))
    if (func.max):
        pop.sort(key=lambda sol: sol.fit, reverse=True)
    else:
        pop.sort(key=lambda sol: sol.fit)
    return best, worst, G, pop[0:k]
    
def timeShortCondition(step, time):
    return time < FIXED_TIME_SHORT
    
def timeShortUpdate(func, pop, step, time):
    best, worst, G = 0, 0, 0
    if func.max:
        best = max(sol.fit for sol in pop)
        worst = min(sol.fit for sol in pop)
    else:
        best = min(sol.fit for sol in pop)
        worst = max(sol.fit for sol in pop)
    G = G_0 * exp(-ALPHA * time/FIXED_TIME_SHORT)
    k = int(ceil(len(pop) * (1 - time/FIXED_TIME_SHORT)))
    if (func.max):
        pop.sort(key=lambda sol: sol.fit, reverse=True)
    else:
        pop.sort(key=lambda sol: sol.fit)
    return best, worst, G, pop[0:k]
    
def timeLongCondition(step, time):
    return time < FIXED_TIME_LONG
    
def timeLongUpdate(func, pop, step, time):
    best, worst, G = 0, 0, 0
    if func.max:
        best = max(sol.fit for sol in pop)
        worst = min(sol.fit for sol in pop)
    else:
        best = min(sol.fit for sol in pop)
        worst = max(sol.fit for sol in pop)
    G = G_0 * exp(-ALPHA * time/FIXED_TIME_LONG)
    k = int(ceil(len(pop) * (1 - time/FIXED_TIME_LONG)))
    if (func.max):
        pop.sort(key=lambda sol: sol.fit, reverse=True)
    else:
        pop.sort(key=lambda sol: sol.fit)
    return best, worst, G, pop[0:k]
    
#takes the function, population size, force calculator, stop condition, seed
def gsa(function, pop_size, force, conditions, columns, outputProgress = True, suffix = "", seed = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    population = initialise(function, pop_size)
    function.getFitnesses(population)

    t = 0
    start = default_timer()
    current_time = default_timer()
    if outputProgress:
        #Track progress
        progColumns = ["t","time"] + ["fitness" + str(i) for i in range(1,pop_size + 1)]
        progress = pd.DataFrame(columns = progColumns)
        progress = progress.append(formatProgress(t, 0, population, progColumns), ignore_index = True)
    while (conditions(t,current_time - start)):
        #Update best, worst, G and kbest
        best, worst, G, kbest = conditions.update(function, population, t, current_time - start)
        
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
        
        current_time = default_timer()
        if (outputProgress and t % OUTPUT_RATE == 0):
            #output current data
            progress = progress.append(formatProgress(t, current_time- start, population, progColumns), ignore_index = True)
    if outputProgress:
        progress.to_csv(function.desc + suffix  + ".csv", index = False)
    return output(columns, best, t, force, function, conditions, current_time - start)
    


def main():
    pass


if __name__ == '__main__':
    main()
else:
    iterationCondition.desc = "fixed-it"
    timeShortCondition.desc = "fixed-time-short"
    timeLongCondition.desc = "fixed-time-long"
    basicForce.desc = "GSA"
    iterationCondition.update = iterationUpdate
    timeShortCondition.update = timeShortUpdate
    timeLongCondition.update = timeLongUpdate
