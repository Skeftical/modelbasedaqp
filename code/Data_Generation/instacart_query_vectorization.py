import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import psycopg2
from psycopg2.extras import NamedTupleCursor
import pandas as pd
from sql_parser.parser import Parser, QueryVectorizer
import numpy as np
import time
import logging
import argparse
from setup import logger
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")
args = parser.parse_args()

if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   formatter = logging.Formatter("[%(asctime)s;%(levelname)s;%(message)s]",
                              "%Y-%m-%d %H:%M:%S")
   handler.setLevel(logging.DEBUG)
   handler.setFormatter(formatter)
   logger.addHandler(handler)
   

queries = []
with open('input/instacart_queries/queries.pkl','rb') as f:
    queries = pickle.load(f)

conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='instacart',user='analyst',password='analyst',cursor_factory=NamedTupleCursor)
cur = conn.cursor()
#Obtain all attribute columns
cur.execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public';")
res = cur.fetchall()
df = pd.DataFrame(res)
print(df)

gattr_to_table_map = { key : value for key,value in zip(df['column_name'].values, df['table_name'].values) }
print(gattr_to_table_map)
#print(attrs_array)
# attrs_dict = { key : [] for key in attrs_array } #dict.fromkeys(attrs_array,[[]]*len(attrs_array))
distinct_attr = {}
i = 0
qdf = None
j = 0
tot_query_answering_time = 0
start = time.time()
for qname,q in queries:
    logger.info("Query :\n{}\n".format(q))
    ####Execute Query and obtain result
    start_query = time.time()
    cur.execute(q)
    tot_query_answering_time+=(time.time()-start_query)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    res_df = res_df.set_index(np.arange(i,i+res_df.shape[0]))
    if res_df.empty:
        logger.debug("Query is empty")
        j+=1
        continue;
    pr = Parser()
    qv = QueryVectorizer(set(df['column_name'].tolist()))
    #Begin parsing the query and vectorizing its parameters
    pr.parse(q)
    dict_obj = pr.get_vector()
    proj_list = pr.get_projections()
    logger.debug("List of Projections : \n {}".format(proj_list))
    rename_names = {key : value  for key in res_df.columns for value in proj_list if value.split('_')[0] in key}
    res_df = res_df.rename(columns=rename_names)
    gattr = pr.get_groupby_attrs()
    logger.debug("List of group-by attributes : \n {}".format(gattr))
    logger.debug("Resulting DataFrame : \n{}".format(res_df))
    #Query Vectorization, adding parameter values and group by values to vector
    for a in dict_obj:
        qv.insert(a, dict_obj[a])
    for g in gattr:
        if g in distinct_attr:
            gattrs = distinct_attr[g]
            logger.debug("Length of dvalues {}".format(len(gattrs)))
            qv.insert(g+'_lb',gattrs)
        else:
            cur.execute("SELECT DISTINCT({0}) FROM {1};".format(g, gattr_to_table_map[g]))
            dvalues = cur.fetchall()
            dvalues = pd.DataFrame(dvalues)[g].tolist()
            qv.insert(g+'_lb',dvalues)
            logger.info("Length of dvalues {}".format(len(dvalues)))
            distinct_attr[g] = dvalues
    #Initializing dataframe to hold the vectors of all queries that will be processed
    if qdf is None:
        qdf = qv.to_dataframe()
    else:
        logger.debug("Current Query Vector dataframe : \n {}".format(qv.to_dataframe().shape))
        qdf = pd.concat([qdf, qv.to_dataframe()], ignore_index=True, sort=False)
    logger.info("Appended QDF holding all queries : \n{}".format(qdf.shape))
    #Concatenating result dataframe with query DataFrame
    #Have to beware of goupby attributes as well as trying to keep the targets
    #in a single column
    if len(gattr)!=0:
        temp = qdf.iloc[i:].merge(res_df, left_on=list(map(lambda x:x+'_lb' ,gattr)), suffixes=('_left','_right'), right_on=gattr,how='left',validate='one_to_one')
        logger.debug("Are the shapes of temp and res_df the same ? : {}".format(temp.shape[0]==res_df.shape[0]))
        try:
            #Index is reset
            temp = temp.set_index(np.arange(i, qdf.shape[0]))
            logger.debug(temp.index)
            qdf.loc[temp.index, proj_list] = temp[list(map(lambda x: '_'.join([x,'right']),proj_list))].values
        except KeyError as e:
            logger.warning("Key of projection in current dataframe does not exist")
            logger.error("The error was : --- {}".format(e))
            qdf = qdf.assign(**{key : np.zeros(qdf.shape[0])*np.nan for key in proj_list})
            qdf.loc[temp.index, proj_list] = temp[proj_list]
    else:#No groupby attributes
        qdf.loc[res_df.index, proj_list] = res_df[proj_list]
    print("Resulting QDF =================")
    print(qdf)
    print(qdf.iloc[i:])

    i=qdf.shape[0]
    j+=1
    logger.info("{}/{} Queries Processed ================".format(j,len(queries)))
with open('catalogues/distinct_attribute_catalogue.pkl', 'wb') as f:
    pickle.dump(distinct_attr, f)
qdf.to_pickle('input/instacart_queries/qdf.pkl')
cur.close()
conn.close()
end = time.time()-start
logger.info("Process took {}".format(end))
logger.info("Total Query Answering Time : {} ({})".format(tot_query_answering_time/end, tot_query_answering_time))
