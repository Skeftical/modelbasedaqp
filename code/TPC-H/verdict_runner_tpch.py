import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re

parser = argparse.ArgumentParser()

parser.add_argument('-s',help='sampling ratio',dest='sampling_ratio')

args = parser.parse_args()

print(args.sampling_ratio)

exit()
if not os.path.exists('../../output/verdict/tpch'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/verdict/tpch')

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('/home/fotis/Desktop/tpch_2_17_0/dbgen/tpch_queries_10/')

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='tpch1g',user='analyst',password='analyst')
#    res = verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.lineitem_x
 #                      FROM public.lineitem SIZE 0.1""")
#    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_x
 #                      FROM public.orders SIZE 0.1""")
 #   verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.partsupp_x
  #                     FROM public.partsupp SIZE 0.1""")
#    print(res)
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    query_names = {}
    i = 0
    regex_lineitem = re.compile(r"lineitem", re.IGNORECASE)
    regex_orders = re.compile(r"orders", re.IGNORECASE)
    regex_order_partsupp = re.compile(r"partsupp", re.IGNORECASE)
    list_of_files = os.listdir(directory)
    for f in list_of_files:
        print(f)
        query_name = os.fsdecode(f).split('.')[0].split('-')[0]
        if query_name not in ['1', '3', '4', '5', '6']:
            continue;
        print("Query Name : {0}".format(os.fsdecode(f).split('.')[0]))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
        sql_query = regex_lineitem.sub("lineitem_x",sql_query)
        sql_query = regex_orders.sub("orders_x",sql_query)
        sql_query = regex_order_partsupp.sub("partsupp_x",sql_query)
        print(sql_query)
        start = time.time()
        try:
            res_df_v = verdict.sql(sql_query)
        except Exception:
            print("Query {} not supported".format(query_name))
            i+=1
            continue;
        end = time.time()-start
        print(res_df_v)
        if 'o_orderdate' in res_df_v.columns:
            res_df_v['o_orderdate'] = res_df_v['o_orderdate'].astype(str)

        res_df_v.to_pickle('../../output/verdict/tpch/{}.pkl'.format(i))

        if query_name not in query_names:
            query_names[query_name] = [i]
        else:
            query_names[query_name].append(i)
        query_answers_dic['time'].append(end)
        query_answers_dic['query_name'].append(query_name)
        i+=1

    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/tpch/query-response-time.csv')
    with open('../../output/verdict/tpch/query-assoc-names.pkl', 'wb') as f:
        pickle.dump(query_names, f)
