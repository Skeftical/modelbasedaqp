import pyverdict
import argparse
import logging
import os
import time
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

if not os.path.exists('../../output/verdict/tpch'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/verdict/tpch')

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('temp')

    verdict = pyverdict.postgres('127.0.0.1',5433,dbname='tpch1g',user='analyst',password='analyst')
    res = verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.lineitem_x
                        FROM public.lineitem SIZE 0.1""")
    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.orders_x
                        FROM public.orders SIZE 0.1""")
    verdict.sql("""CREATE SCRAMBLE IF NOT EXISTS public.partsupp_x
                        FROM public.partsupp SIZE 0.1""")
    print(res)
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    for f in os.listdir(directory):
        print(f)
        print("Query Name : {0}".format(os.fsdecode(f).split('.')[0]))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
            start = time.time()
            res_df_v = verdict.sql(sql_query)
            end = time.time()-start
            res_df_v.to_pickle('../../output/verdict/tpch/{}.pkl'.format(query_name))
            query_answers_dic['time'].append(end)
            query_answers_dic['query_name'].append(query_name)
    verdict.close()
    qa = pd.DataFrame(query_answers_dic)
    qa.to_csv('../../output/verdict/query-response-time.csv')
