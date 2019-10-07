import psycopg2
import argparse
import logging
import os
import pandas as pd
import logging
import time
import pickle
import sys
from psycopg2.extras import NamedTupleCursor

# logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
#                      action="store_true")
#parser.add_argument('--pass',dest='pass', help='pass connection password')
parser.add_argument("password")
args = parser.parse_args()# parser.add_argument('source')
#
# if args.verbosity:
#     print("verbosity turned on")
#     handler = logging.StreamHandler(sys.stdout)
#     handler.setLevel(logging.DEBUG)
#     logger.addHandler(handler)
# #
# print(args.source)
if not os.path.exists('../../output/backend-redshift/instacart'):
        print('creating directory Accuracy')
        os.makedirs('../../output/backend-redshift/instacart')

if __name__=='__main__':
    THRESH = 60000*5 #minutes
    print("main executing")
    with open('../../input/instacart_queries/queries-test.pkl', 'rb') as f:
        queries = pickle.load(f)
    conn = psycopg2.connect(host='examplecluster.ck9mym5op4yd.eu-west-1.redshift.amazonaws.com',port=5439,dbname='dev',user='awsuser',password=args.password,cursor_factory=NamedTupleCursor,options='-c statement_timeout={}'.format(THRESH) )
    cur = conn.cursor()
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    i = 0
    sys.stdout.flush()
    for qname,q in queries:
        print("Query {}".format(q))
        start = time.time()
        try:
            cur.execute(q)
            res = cur.fetchall()
            end = time.time()-start
        except psycopg2.errors.QueryCanceled as e:
            print("Query {} exceeded threshold".format(qname))
            print(e)
            end = THRESH/1000
            conn.rollback()
            sys.stdout.flush()
        query_answers_dic['time'].append(end)
        query_answers_dic['query_name'].append(qname)
        i+=1
      
    cur.close()
    conn.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/backend-redshift/instacart/query-response-time.csv')
