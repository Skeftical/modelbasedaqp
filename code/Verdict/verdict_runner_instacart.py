import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
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

if not os.path.exists('../../output/verdict/instacart'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/verdict/instacart')

if __name__=='__main__':
    print("main executing")
    with open('../../input/instacart_queries/queries-test.pkl', 'rb') as f:
       queries = pickle.load(f)

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='instacart',user='analyst',password='analyst')
#    res = verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.lineitem_x
#                        FROM public.lineitem SIZE 0.1""")
    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.order_products_instacart_x
                       FROM public.order_products SIZE 0.1""")
    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_instacart_x
                       FROM public.orders SIZE 0.1""")
#    print(res)
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    for i,q in enumerate(queries):
            start = time.time()
            try:
                res_df_v = verdict.sql(q)
            except Exception:
                print("Query {} not supported".format(i))
            end = time.time()-start
            res_df_v.to_pickle('../../output/verdict/instacart/{}.pkl'.format(i))
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(i)
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/instacart/query-response-time.csv')
