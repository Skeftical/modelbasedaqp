import pyverdict
import argparse
import logging
import os
import sys
import time
import pandas as pd
import pickle
import re
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')

if not os.path.exists('output/performance'):
        print('creating ', 'performance')
        os.makedirs('output/performance')
if not os.path.exists('output/performance/csvs'):
        print('creating ', 'performance csvs')
        os.makedirs('output/performance/csvs')

if __name__=='__main__':
    print("main executing")

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='tpch1g',user='analyst',password='analyst')

    verdict.sql("""DROP SCRAMBLE public.lineitem_x
                     ON public.lineitem """)
    verdict.sql("""DROP SCRAMBLE public.orders_x
                     ON public.orders""")
    verdict.sql("""DROP SCRAMBLE public.partsupp_x
                     ON public.partsupp""")

    result = {}
    result['sample_size'] = []
    result['time'] = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    start = time.time()
    for ratio in ratios:
        verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.lineitem_x
                         FROM public.lineitem SIZE {}""".format(ratio))
        verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_x
                         FROM public.orders SIZE {}""".format(ratio))
        verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.partsupp_x
                              FROM public.partsupp SIZE {}""".format(ratio))
        end = time.time()-start
        result['sample_size'].append(ratio)
        result['time'].append(end)
        verdict.sql("""DROP SCRAMBLE public.lineitem_x
                         ON public.lineitem """)
        verdict.sql("""DROP SCRAMBLE public.orders_x
                         ON public.orders""")
        verdict.sql("""DROP SCRAMBLE public.partsupp_x
                         ON public.partsupp""")

    result = pd.DataFrame(result)
    result.to_csv('../../output/performance/csvs/verdict/verdict-sample-building-ratio.csv')

#    print(res)
