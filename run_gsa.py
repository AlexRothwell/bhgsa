import pandas as pd
import gsa
import barneshut as bh
import functions
from cec2013lsgo.cec2013 import Benchmark

def main():
    dims = 30
    pop_size = 50
    num_funcs = 23
    forces = [gsa.basicForce, bh.BHForce(1e5), bh.BHForce(0.5), bh.BHForce(1e-5), bh.BHForce(0)]
    columns = ["FUNCTION","SOLVER","BEST","ITERATIONS","TIME"]
    repetitions = 20
    
    condition = gsa.iterationCondition
    
    result = pd.DataFrame(columns)

    #Run 23 original functions
    for prob_num in range(12,num_funcs + 1):
        func = functions.getFunction(prob_num,dims)
        for force in forces:
            func.desc = str(prob_num) + "-" + force.desc
            print("Processing problem {0} with {1}".format(prob_num, force.desc))
            for i in range(repetitions):
                result = result.append(gsa.gsa(func,pop_size,force,condition, columns, outputProgress=True), ignore_index = True)
        
    result.to_csv("out-" + condition.desc + ".csv", index = False)
    
    #Run 15 CEC2013 functions
    result = pd.DataFrame(columns = columns)

    for prob_num in range(2,3):#Benchmark().get_num_functions() + 1):
        info = Benchmark().get_info(prob_num)
        dims = info['dimension']
        cec_func = Benchmark().get_function(prob_num)
        func = functions.FunctionMaker("cec-" + str(prob_num), [info['upper']]*dims, [info['lower']]*dims, dims, 
                                       False, lambda sol: cec_func(sol))
        for force in forces:
            func.desc = "cec-" + str(prob_num) + "-" + force.desc
            print("Processing problem {0} with {1}".format(prob_num, force.desc))
            for i in range(repetitions):
                result = result.append(gsa.gsa(func,pop_size,force,condition, columns, outputProgress=True), ignore_index = True)
        
    result.to_csv("cec-" + condition.desc + ".csv", index = False)
    

if __name__ == '__main__':
    main()