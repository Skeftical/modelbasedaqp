import psycopg2
import argparse
import logging
import os
import pandas as pd
import logging
import time
#
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")
parser.add_argument('source')
args = parser.parse_args()

if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   logger.addHandler(handler)

source = args.source
dbname = args.dbname
if not os.path.exists('../../output/backend-postgres-actual'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/backend-postgres-actual')

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode(source)
    conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname=dbname,user='analyst',password='analyst')
    cur = conn.cursor()
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    for f in os.listdir(directory):
        print(f)
        query_name = os.fsdecode(f).split('.')[0]
        print("Query Name : {0}".format(query_name))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
            # print(sql_query)
            start = time.time()
            cur.execute(sql_query)
            res = cur.fetchall()
            end = time.time()-start
            res_df = pd.DataFrame(res)
            res_df.to_pickle('../../output/backend-postgres-actual/{}/{}.pkl'.format(dbname, query_name))
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(query_name)
    cur.close()
    conn.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/backend-postgres-actual/query-response-time.csv')
