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


if not os.path.exists('../../output/verdict/crimes'):
        os.makedirs('../../output/verdict/crimes')

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
    test_queries = load_data()
    print(test_queries.head(5))
    # res = verdict.sql("SELECT DISTINCT(primary_type) FROM crimes;")
    # print(res)

    # verdict.sql("DROP ALL SCRAMBLE public.lineitem;")
    # verdict.sql("DROP ALL SCRAMBLE public.orders;")
    # verdict.sql("DROP ALL SCRAMBLE public.partsupp;")
    # res = verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.lineitem_x
    #                   FROM public.lineitem SIZE {}""".format(sampling_ratio))
    # verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_x
    #                   FROM public.orders SIZE {}""".format(sampling_ratio))
    # verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.partsupp_x
    #                   FROM public.partsupp SIZE {}""".format(sampling_ratio))


    # query_answers_dic = {}
    # query_answers_dic['query_name'] = []
    # query_answers_dic['time'] = []
    # query_names = {}
    # i = 0
    # regex_lineitem = re.compile(r"lineitem", re.IGNORECASE)
    # regex_orders = re.compile(r"orders", re.IGNORECASE)
    # regex_order_partsupp = re.compile(r"partsupp", re.IGNORECASE)
    # list_of_files = os.listdir(directory)
    # for f in list_of_files:
    #     print(f)
    #     query_name = os.fsdecode(f).split('.')[0].split('-')[0]
    #     if query_name not in ['1', '3', '4', '5', '6']:
    #         continue;
    #     print("Query Name : {0}".format(os.fsdecode(f).split('.')[0]))
    #     with open(os.path.join(directory,f),"r") as sql_query_file:
    #         sql_query = sql_query_file.read()
    #     sql_query = regex_lineitem.sub("lineitem_x",sql_query)
    #     sql_query = regex_orders.sub("orders_x",sql_query)
    #     sql_query = regex_order_partsupp.sub("partsupp_x",sql_query)
    #     print(sql_query)
    #     start = time.time()
    #     try:
    #         res_df_v = verdict.sql(sql_query)
    #     except Exception:
    #         print("Query {} not supported".format(query_name))
    #         i+=1
    #         continue;
    #     end = time.time()-start
    #     print(res_df_v)
    #     if 'o_orderdate' in res_df_v.columns:
    #         res_df_v['o_orderdate'] = res_df_v['o_orderdate'].astype(str)
    #
    #     res_df_v.to_pickle('../../output/verdict/tpch-{}/{}.pkl'.format(sampling_ratio,i))
    #
    #     if query_name not in query_names:
    #         query_names[query_name] = [i]
    #     else:
    #         query_names[query_name].append(i)
    #     query_answers_dic['time'].append(end)
    #     query_answers_dic['query_name'].append(query_name)
    #     i+=1
    #
    # verdict.close()
    # qa = pd.DataFrame(query_answers_dic)
    # qa.to_csv('../../output/verdict/tpch-{}/query-response-time.csv'.format(sampling_ratio))
    # with open('../../output/verdict/tpch-{}/query-assoc-names.pkl'.format(sampling_ratio), 'wb') as f:
    #     pickle.dump(query_names, f)
