import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re
from sklearn import metrics
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-s','-sampling_ratio',help='sampling ratio',dest='sampling_ratio',required=True)

args = parser.parse_args()

print(args.sampling_ratio)
sampling_ratio = args.sampling_ratio

if not os.path.exists('../../output/verdict/crimes-{}'.format(sampling_ratio)):
        os.makedirs('../../output/verdict/crimes-{}'.format(sampling_ratio))

def load_data():
    print("Loading Data...")

    test_df = pd.read_csv('../../input/Crimes_Workload/test_workload_gauss-5-users-50000.csv', header=0, index_col=0)
    test_df.reset_index(drop=True, inplace=True)

    test_df = test_df.replace([np.inf,-np.inf], np.nan).dropna()

    test_df['x_l'] = test_df['x']-test_df['x_range']
    test_df['x_h'] = test_df['x']+test_df['x_range']
    test_df['y_l'] = test_df['y']-test_df['y_range']
    test_df['y_h'] = test_df['y']+test_df['y_range']
    test_df = test_df.drop(['x','y','x_range','y_range'],axis=1)

    return test_df

if __name__=='__main__':
    print("main executing")

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='postgres',user='analyst',password='analyst')
    #Prepare Samples
    verdict.sql("DROP ALL SCRAMBLE public.crimes;")
    res = verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.crimes_x
                      FROM public.crimes SIZE {}""".format(sampling_ratio))
    print(res)
    #Evaluate on queries
    test_queries = load_data()
    query_answers_dic = {}
    query_answers_dic['y_count'] = []
    query_answers_dic['y_hat_count'] = []
    query_answers_dic['y_sum'] = []
    query_answers_dic['y_hat_sum'] = []
    query_answers_dic['y_avg'] = []
    query_answers_dic['y_hat_avg'] = []

    for tup in test_queries.iterrows():
        y_count, y_sum, y_avg, x_l, x_h, y_l, y_h = tup[1]
        res = verdict.sql("""
            SELECT COUNT(*), SUM(arrest), AVG(beat)
            FROM crimes_x
            WHERE x_coordinate>={} AND x_coordinate<={}
            AND y_coordinate>={}   AND y_coordinate<={}
        """.format(x_l, x_h, y_l, y_h))
        y_hat_count = float(np.log(res['c2'].values.astype(float)))
        y_hat_sum = float(np.log(res['s3'].values.astype(float)))
        y_hat_avg = float(res['a4'].values.astype(float))

        query_answers_dic['y_count'].append(y_count)
        query_answers_dic['y_hat_count'].append(y_hat_count)
        query_answers_dic['y_sum'].append(y_sum)
        query_answers_dic['y_hat_sum'].append(y_hat_sum)
        query_answers_dic['y_avg'].append(y_avg)
        query_answers_dic['y_hat_avg'].append(y_hat_avg)
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/crimes-{}/predictions-answers.csv'.format(sampling_ratio))
