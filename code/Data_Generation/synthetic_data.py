import pandas as pd
import numpy as np
import os
import sys
os.chdir("../../")
print(os.listdir('.'))
sys.path.append(".")
from confs import Config
import datetime
import logging
import re
logging.basicConfig(stream=sys.stdout, level=logging.ERROR,)
logger = logging.getLogger(__name__)

SELECTIVITY = 0.1 # Selectivity scaler
MIN_VAL = 0
MAX_VAL = 1

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def generate_dataset(min_val=MIN_VAL, max_val=MAX_VAL, total=Config.data):
    '''
    Generate Uniform Dataset across values

    Parameters:
    ==============
    min_val : float
        minimum value in columns
    max_val : float
        maximum value in columns
    total : int
        total number of elements in dataset
    '''
    if not os.path.exists('input'):
        print('creating ', 'input')
        os.makedirs('input')
    if not os.path.exists('input/synthetic_data'):
        print('creating ' ,'synethtic_data')
        os.makedirs('input/synthetic_data')

    existing = set(os.listdir('input/synthetic_data'))

    columns = Config.columns
    for col in columns:
        if "data-{}.csv".format(col) not in existing:
            a = np.random.uniform(min_val, max_val,(total, col))
            print('Creating data with columns : %d' % col)
            np.savetxt("input/synthetic_data/data-{}.csv".format(col), a, delimiter=",")
        else:
            print('Data with columns %d already exists' % col)

def return_query(num_of_predicates, array,min_val=MIN_VAL, max_val=MAX_VAL):
    '''
    Return query with predicates at random columns

    Parameters:
    =============
    num_of_predicates : int
        Number of predicates to generate for the query
    array : array[int]
        Array holding the indices of columns
    '''
    # p = np.random.uniform(min_val,max_val,(1,num_of_predicates)) #Random center point
    p = np.random.normal((max_val-min_val)/2., 0.01 ,(1,num_of_predicates))
    temp = np.zeros((1,len(array)))
    for idx,predicate in np.ndenumerate(p):
        pos = array.pop(np.random.randint(0,len(array)))
        temp[:,pos] = predicate

    return temp


# """
# Transformer functions for turning vectors to predicates with a pre-defined selectivity
# Range is obtained by the width of a bin in a histogram, since data is uniform
# queries will have similar selectivities
# np.vectorize applies the function over all elements of vectors ie creates mapping
# """
# def transformer_low(a):
#     if a!=0:
#         r = float(np.random.normal(SELECTIVITY**(1/10), 0.01*(MAX_VAL-MIN_VAL) ,1 ))/2.0
#         logger.info("r in lower vector {0}".format(a-r))
#         return a-r
#     else:
#         return MIN_VAL
# def transformer_high(a):
#     if a!=0:
#         r = float(np.random.normal(SELECTIVITY**(1/10), 0.01*(MAX_VAL-MIN_VAL) ,1 ))/2.0
#         logger.info("r in higher vector {0}".format(a+r))
#         return a+r
#     else:
#         return MAX_VAL
# vfunc_low = np.vectorize(transformer_low)
# vfunc_high = np.vectorize(transformer_high)


def generate_queries(queries=Config.queries):
    '''
    Generate workloads for a pre-specified amount of Queries
    Arrays of columns and predicates are pre-defined inside the function
    please see source code.
    '''
    columns = Config.columns
    predicates = Config.predicates
    if not os.path.exists('input/synthetic_workloads'):
        print('creating directory' ,'synthetic_workloads')
        os.makedirs('input/synthetic_workloads')

    if not os.path.exists('input/synthetic_workloads/{}-Queries'.format(queries)):
        print('creating directory' ,'{}-Queries'.format(queries))
        os.makedirs('input/synthetic_workloads/{}-Queries'.format(queries))
    existing = set(os.listdir('input/synthetic_workloads/{}-Queries'.format(queries)))

    for COLS in columns:
        try:
            a = np.loadtxt('input/synthetic_data/data-{0}.csv'.format(COLS),delimiter=',')
        except OSError:
            print("File data-%d not found; exiting" %COLS)
            sys.exit(0)

        for PREDICATES in predicates:
            #Continue only if predicates are less than COLS
            if PREDICATES>COLS:
                continue;
            query_set = []

            if  any(filter(lambda x : re.match("query-workload-predicates_{}-cols_{}-.*.csv".format(PREDICATES, COLS),x),existing)) :
                print("Query workload with : {} predicates and {} cols already exists; skipping ".format(PREDICATES, COLS))
                continue;
            print("Generating Query workload with {0} columns and {1} predicates".format(COLS, PREDICATES))
            errors = 0
            for i in range(queries):
                array = list(range(0, COLS))
                p = return_query(PREDICATES, array)
                logger.info("position vector {0}".format(str(p)))
                # print("Vector is:\n")
                # print(p)
                r = float(np.random.normal(SELECTIVITY**(1/COLS), 0.01*(MAX_VAL-MIN_VAL) ,1 ))/2.0
                p_l = np.maximum(np.zeros(p.shape),(p-r)*np.ceil(p))
                # print("Lower Bound \n")
                # print(p_l)
                p_h = np.minimum(np.ones(p.shape),np.abs((p+r)*np.ceil(p)+(np.ceil(p)-np.ones(p.shape))))
                logger.info("lower bound vector {0}".format(str(p_l)))
                logger.info("higher bound vector {0}".format(str(p_h)))
                assert np.all((p_l>=MIN_VAL) & (p_l<MAX_VAL)) and np.all((p_h<=MAX_VAL) & (p_h>MIN_VAL))
                # print("High bound \n")
                # print(p_h)
                result_set = a[(a>p_l).all(axis=1) & (a<p_h).all(axis=1),:]

                count = result_set.shape[0]
                if count==0:
                    mean = np.nan
                    sum_ = np.nan
                    min_ = np.nan
                    max_ = np.nan
                    errors+=1
                else:
                    mean = np.mean(result_set[:,0])
                    sum_ = np.sum(result_set[:,0])
                    min_ = np.min(result_set[:,0])
                    max_ = np.max(result_set[:,0])
                query_set.append(np.column_stack((p_l, p_h, count, sum_, mean, min_, max_)))
                printProgressBar(i, queries,prefix = 'Progress:', suffix = 'Complete', length = 50)
            print(" Fraction of queries with nans {}".format(float(errors)/queries))
            q_set = np.array(query_set)

            np.savetxt("input/synthetic_workloads/{}-Queries/query-workload-predicates_{}-cols_{}.csv".format(queries,PREDICATES, COLS), q_set.reshape(queries,q_set.shape[2]), delimiter=",")

if __name__=='__main__':
    np.random.seed(0)
    generate_dataset()
    generate_queries()
