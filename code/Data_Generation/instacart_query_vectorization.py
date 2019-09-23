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
afs = {}
distinct_attr = {}
i = 0
qdf = None
j = 0
start = time.time()
for qname,q in queries:
    print(q)
    cur.execute(q)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    res_df = res_df.set_index(np.arange(i,i+res_df.shape[0]))
    print(res_df)
    if res_df.empty:
        continue;
    pr = Parser()
    qv = QueryVectorizer(set(df['column_name'].tolist()))

    pr.parse(q)
    dict_obj = pr.get_vector()
    proj_list = pr.get_projections()
    print(proj_list)
    print({key : value  for key,value in zip(res_df.columns, proj_list) if key in value})
    gattr = pr.get_groupby_attrs()
    print(gattr)
    for a in dict_obj:
        qv.insert(a, dict_obj[a])
    for g in gattr:
        if g in distinct_attr:
            gattrs = distinct_attr[g]
        else:
            cur.execute("SELECT DISTINCT({0}) FROM {1};".format(g, gattr_to_table_map[g]))
            dvalues = cur.fetchall()
            dvalues = pd.DataFrame(dvalues)[g].tolist()
            qv.insert(g+'_lb',dvalues)
            distinct_attr[g] = dvalues
    if qdf is None:
        qdf = qv.to_dataframe()
    else:
        print(qdf.shape)
        print(qv.to_dataframe().shape)
        qdf = pd.concat([qdf, qv.to_dataframe()], ignore_index=True, sort=False)
    print(qdf.shape)
    if len(gattr)!=0:
        temp = qdf[i:i+res_df.shape[0]].merge(res_df, left_on=list(map(lambda x:x+'_lb' ,gattr)), right_on=gattr,how='left',suffixes=('_left_{}'.format(j),'_right_{}'.format(j)))
        qdf = pd.concat([qdf.iloc[:i],temp],ignore_index=True, sort=False)
    else:#No groupby attributes
        qdf = qdf.merge(res_df, left_index=True, right_index=True, how='left',suffixes=('_left_{}'.format(j),'_right_{}'.format(j)))
    qdf = qdf.drop(columns=gattr)
    print(qdf)
    for af in proj_list:
        if af in afs:
            afs[af].append((i,qdf.shape[0]))
        else:
            afs[af] = [(i,qdf.shape[0])]
    print(afs)
    i=qdf.shape[0]
    j+=1
    print("{}/{} Queries Processed ================".format(j,len(queries)))
with open('input/instacart_queries-/afs.pkl','wb') as f:
    pickle.dump(afs, f)
with open('catalogues/distinct_attribute_catalogue.pkl', 'wb') as f:
    pickle.dump(distinct_attr, f)
qdf.to_pickle('input/instacart_queries/qdf.pkl')
cur.close()
conn.close()
print("Process took {}".format(time.time()-start))
