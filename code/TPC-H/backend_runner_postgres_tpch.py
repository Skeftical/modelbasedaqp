import psycopg2
import argparse
import logging
import os
import pandas as pd
import logging
import time
import sys
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")
args = parser.parse_args()

if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   logger.addHandler(handler)


if not os.path.exists('../../output/backend-postgres-actual/tpch'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/backend-postgres-actual/tpch')

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('/home/fotis/Desktop/tpch_2_17_0/dbgen/tpch_queries_10/')
    conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='tpch1g',user='analyst',password='analyst')
    cur = conn.cursor()
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    query_names = {}
    i = 0
    for f in os.listdir(directory):
        query_name = os.fsdecode(f).split('.')[0].split('-')[0]
        if query_name not in ['1', '3', '4', '5', '6']:
            continue;
        print("Query Name : {0}".format(query_name))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
            # print(sql_query)
            start = time.time()
            cur.execute(sql_query)
            res = cur.fetchall()
            end = time.time()-start
            res_df = pd.DataFrame(res)
            res_df.to_pickle('../../output/backend-postgres-actual/tpch/{}.pkl'.format(query_name))
            if query_name not in query_names:
                query_names[query_name] = [i]
            else:
                query_names[query_name].append(i)
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(query_name)
            i+=1
    cur.close()
    conn.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/backend-postgres-actual/tpch/query-response-time.csv')
    with open('../../output/backend-postgres-actual/tpch/query-assoc-names.pkl', 'wb') as f:
        pickle.dump(query_names, f)
