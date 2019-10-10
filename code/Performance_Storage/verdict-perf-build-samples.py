import pyverdict
import argparse
import logging
import os
import time
import pandas as pd
import pickle
import re


if not os.path.exists('output/performance'):
        print('creating ', 'performance')
        os.makedirs('output/performance')
if not os.path.exists('output/performance/csvs'):
        print('creating ', 'performance csvs')
        os.makedirs('output/performance/csvs')

if __name__=='__main__':
    print("main executing")

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='tpch1g',user='analyst',password='analyst')
    result = {}
    result['sample_size'] = []
    result['time'] = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    start = time.time()
    for ratio in ratios:
        verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.order_products_instacart_x
                         FROM public.order_products SIZE {}""".format(ratio))
        verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_instacart_x
                         FROM public.orders SIZE {}""".format(ratio))
        end = time.time()-start
        result['sample_size'].append(ratio)
        result['time'].append(end)
        verdict.sql("""DROP SCRAMBLE public.order_products_instacart_x
                         ON public.order_products """)
        verdict.sql("""DROP SCRAMBLE public.orders_instacart_x
                         ON public.orders""".format(ratio))

    resukt = pd.DataFrame(result)
    qa.to_csv('../../output/performance/csvs/verdict/verdict-sample-building-ratio.csv')

#    print(res)
