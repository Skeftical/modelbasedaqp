import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re

parser = argparse.ArgumentParser()

parser.add_argument('-s','-sampling_ratio',help='sampling ratio',dest='sampling_ratio',required=True)

args = parser.parse_args()

print(args.sampling_ratio)
sampling_ratio = args.sampling_ratio


if not os.path.exists('../../output/verdict/instacart-1000-{}'.format(sampling_ratio)):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/verdict/instacart-1000-{}'.format(sampling_ratio))

if __name__=='__main__':
    print("main executing")
    with open('../../input/instacart_queries/queries-test-1000.pkl', 'rb') as f:
       queries = pickle.load(f)

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='instacart',user='analyst',password='analyst')
    verdict.sql("DROP ALL SCRAMBLE public.order_products")
    verdict.sql("DROP ALL SCRAMBLE public.orders")

    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.order_products_instacart_x
                   FROM public.order_products SIZE {}""".format(sampling_ratio))
    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_instacart_x
                   FROM public.orders SIZE {}""".format(sampling_ratio))
#    print(res)
#    sys.exit(0)
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
            res_df_v.to_pickle('../../output/verdict/instacart-1000-{}/{}.pkl'.format(sampling_ratio,i))
            if qname not in query_names:
                query_names[qname] = [i]
            else:
                query_names[qname].append(i)
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(qname)
            i+=1
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/instacart-1000-{}/query-response-time.csv'.format(sampling_ratio))
    with open('../../output/verdict/instacart-1000-{}/query-assoc-names.pkl'.format(sampling_ratio), 'wb') as f:
        pickle.dump(query_names, f)
