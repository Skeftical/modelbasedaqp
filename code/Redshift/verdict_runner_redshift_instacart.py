import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re

parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
#                      action="store_true")
#parser.add_argument('--pass',dest='pass', help='pass connection password')
parser.add_argument("password")
args = parser.parse_args()# parser.add_argument('source')
#
# if args.verbosity:
#    print("verbosity turned on")
#    handler = logging.StreamHandler(sys.stdout)
#    handler.setLevel(logging.DEBUG)
#    logger.addHandler(handler)
#
# print(args.source)

if not os.path.exists('../../output/verdict-redshift/instacart'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/verdict-redshift/instacart')

if __name__=='__main__':
    print("main executing")
    with open('../../input/instacart_queries/queries-test.pkl', 'rb') as f:
       queries = pickle.load(f)

    verdict = pyverdict.redshift(host='examplecluster.ck9mym5op4yd.eu-west-1.redshift.amazonaws.com',port=5439,dbname='dev',user='awsuser',password=args.pass)

#    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.order_products_instacart_x
 #                     FROM public.order_products SIZE 0.1""")
 #   verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_instacart_x
  #                    FROM public.orders SIZE 0.1""")
#    print(res)
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    query_names = {}
    i = 0
    regex_orders = re.compile(r"orders", re.IGNORECASE)
    regex_order_products = re.compile(r"order_products", re.IGNORECASE)
    for qname,q in queries:
            print(q)
            q = regex_orders.sub("orders_instacart_x",q)
            q = regex_order_products.sub("order_products_instacart_x",q)
            print("Changed Query :")
            print(q)
            print("================================")
            start = time.time()
            try:
                res_df_v = verdict.sql(q)
            except Exception as e:
                print("Query {} not supported".format(qname))
                print(e)
            end = time.time()-start
            res_df_v.to_pickle('../../output/verdict/instacart/{}.pkl'.format(i))
            if qname not in query_names:
                query_names[qname] = [i]
            else:
                query_names[qname].append(i)
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(qname)
            i+=1
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/instacart/query-response-time.csv')
    with open('../../output/verdict/instacart/query-assoc-names.pkl', 'wb') as f:
        pickle.dump(query_names, f)
