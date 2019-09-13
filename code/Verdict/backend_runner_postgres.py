import psycopg2
import argparse
import logging
import os
import pandas as pd
import logging
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
#                     action="store_true")
# parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")
# parser.add_argument('source')
# args = parser.parse_args()
#
# if args.verbosity:
#    print("verbosity turned on")
#    handler = logging.StreamHandler(sys.stdout)
#    handler.setLevel(logging.DEBUG)
#    logger.addHandler(handler)
#
# print(args.source)
if not os.path.exists('../../output/backend-postgres-actual'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../ooutput/backend-postgres-actual')

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('temp')
    conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='tpch1g',user='analyst',password='analyst')
    cur = conn.cursor()
    query_answers_dic = {}
    query_answers_dic['time'] = []
    for f in os.listdir(directory):
        print(f)
        query_name = os.fsdecode(f).split('.')[0]
        print("Query Name : {0}".format(query_name))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
            print(sql_query)
            cur.execute(sql_query)
            res = cur.fetchall()
            res_df = pd.DataFrame(res)
            res_df.to_pickle('../../output/backend-postgres-actual/{}.pkl'.format(query_name))
    cur.close()
    conn.close()
