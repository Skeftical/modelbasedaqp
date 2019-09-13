import psycopg2
import argparse
import logging
import os
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

if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('temp')
    conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='tpch1g',user='analyst',password='analyst')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM lineitem;")
    print(cur.fetchall())
    query_answers_dic = {}
    query_answers_dic['query'] = []
    query_answers_dic['result'] = []
    query_answers_dic['time'] = []
    # for f in os.listdir(directory):
    #     print(f)
    #     print("Query Name : {0}".format(os.fsdecode(f).split('.')[0]))
    #     with open(os.path.join(directory,f),"r") as sql_query_file:
    #         sql_query = sql_query_file.read()
    #         print(sql_query)
    #         res_df_v = verdict.sql(sql_query)
    #         print(res_df_v)
    #         res = verdict.sql("SELECT avg(l_extendedprice) FROM lineitem;")
    #         print(res)
    cur.close()
    conn.close()
