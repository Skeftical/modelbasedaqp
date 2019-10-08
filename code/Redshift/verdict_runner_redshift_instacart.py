import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re
import signal

class CustomException(Exception):
    pass

def handler(signum, frame):
        raise CustomException("end of time")
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
    signal.signal(signal.SIGALRM, handler)
    THRESH = 60000*5 #minutes
    verdict = pyverdict.redshift(host='examplecluster.ck9mym5op4yd.eu-west-1.redshift.amazonaws.com',port=5439,dbname='dev',user='awsuser',password=args.password)
    start = time.time()
#    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.order_products_instacart_x
#                     FROM public.order_products SIZE 0.1""")
#    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_instacart_x
#                     FROM public.orders SIZE 0.1""")
    print("Time to build samples {}".format(time.time()-start))
#    print(res)
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    i = 0
    regex_orders = re.compile(r"orders", re.IGNORECASE)
    regex_order_products = re.compile(r"order_products", re.IGNORECASE)
    counter = {}
    for qname,q in queries:
            print(q)
            counter[qname] = counter.get(qname,0)+1
            if counter[qname]>=5:
                continue;
            q = regex_orders.sub("orders_instacart_x",q)
            q = regex_order_products.sub("order_products_instacart_x",q)
            print("Changed Query :")
            print(q)
            print("================================")
            start = time.time()
            try:
                signal.alarm(THRESH//1000)
                res_df_v = verdict.sql(q)
                end = time.time()-start
            except CustomException as e:
                print("Query {} timed outd".format(qname))
                print(e)
                end = THRESH/1000
            except Exception as e:
                print("Query {} not supported".format(qname))
                print(e)
                end = THRESH/1000

            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(qname)
            i+=1
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict-redshift/instacart/query-response-time.csv')
