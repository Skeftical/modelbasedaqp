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

queries = []
with open('input/instacart_queries/queries.pkl','rb') as f:
    queries = pickle.load(f)
pr = Parser()
conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='instacart',user='analyst',password='analyst',cursor_factory=NamedTupleCursor)
cur = conn.cursor()
#Obtain all attribute columns
cur.execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public';")
res = cur.fetchall()
df = pd.DataFrame(res)
print(df)
# attrs_array = []
# for a in df['column_name'].values:
# 	attrs_array.append('_'.join([a,'lb']))
# 	attrs_array.append('_'.join([a,'ub']))
gattr_to_table_map = { key : value for key,value in zip(df['column_name'].values, df['table_name'].values) }
print(gattr_to_table_map)
print(attrs_array)
# attrs_dict = { key : [] for key in attrs_array } #dict.fromkeys(attrs_array,[[]]*len(attrs_array))
afs = {}
distinct_attr = {}
qv = QueryVectorizer(df['column_name'].tolist())

for i,q in enumerate(queries):
    print(q)
    pr.parse(q)
    dict_obj = pr.get_vector()


    proj_dict = pr.get_projections()
    for af in proj_dict:
        if af in afs:
            afs[af].append(i)
        else:
            afs[af] = [i]
    print(afs)
    gattr_dict = pr.get_groupby_attrs()
    for g in gattr_dict:
        if g in distinct_attr:
            gattrs = distinct_attr[g]
        else:
            cur.execute("SELECT DISTINCT({0}) FROM {1};".format(g, gattr_to_table_map[g]))
            dvalues = cur.fetchall()
            dvalues = pd.DataFrame(dvalues)[g].tolist()
            qv.insert(g+'_lb',dvalues)
    for a in dict_obj:
        qv.insert(a, dict_obj[a])
    print(qv.to_dataframe())
    cur.execute(q)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    print(res_df)
    break;
cur.close()
conn.close()
