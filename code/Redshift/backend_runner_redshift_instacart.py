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
# parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
#                      action="store_true")
parser.add_argument('--pass',dest='pass' help='pass connection password')
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
        logger.info('creating directory Accuracy')
        os.makedirs('../../output/backend-redshift/instacart')

if __name__=='__main__':
    print("main executing")
    with open('../../input/instacart_queries/queries-test.pkl', 'rb') as f:
        queries = pickle.load(f)
    conn = psycopg2.connect(host='examplecluster.ck9mym5op4yd.eu-west-1.redshift.amazonaws.com',port=5439,dbname='dev',user='awsuser',password=args.pass,cursor_factory=NamedTupleCursor)

    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    query_names = {}
    i = 0
    for qname,q in queries:
        print("Query {}".format(q))
        start = time.time()
        cur.execute(q)
        res = cur.fetchall()
        end = time.time()-start
        res_df = pd.DataFrame(res)
        res_df.to_pickle('../../output/backend-redshift/instacart/{}.pkl'.format(i))
        if qname not in query_names:
            query_names[qname] = [i]
        else:
            query_names[qname].append(i)
        query_answers_dic['time'].append(end)
        query_answers_dic['query_name'].append(qname)
        i+=1
    cur.close()
    conn.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/backend-redshift/instacart/query-response-time.csv')
    with open('../../output/backend-redshift/instacart/query-assoc-names.pkl', 'wb') as f:
        pickle.dump(query_names, f)
