import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import psycopg2
from psycopg2.extras import NamedTupleCursor
import pandas as pd
from sql_parser.parser import Parser

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
set_of_attributes = set(map(lambda x: x[0].replace('_',''),df['column_name']))
attrs_array = []
for a in set_of_attributes:
	attrs_array.append('_'.join([a,'lb']))
	attrs_array.append('_'.join([a,'ub']))
print(attrs_array)
attrs_dict = { key : [] for key in attrs_array } #dict.fromkeys(attrs_array,[[]]*len(attrs_array))
afs = {}
distinct_attr = {}
sys.exit(0)
for i,q in enumerate(queries):
    print(q)
    pr.parse(q)



    dict_obj = pr.get_vector()
    for a in attrs_dict:
        if a=='af':
             continue;
        attr, dire = a.split('_')
        if attr not in dict_obj:#If this attribute is not in the query vector then leave as None
            attrs_dict[a].append(None) # Fill with None initially
        else:
            if dire=='lb':
                lb = dict_obj[attr].get('lb',None)
                attrs_dict[a].append(lb)
            else:
                ub = dict_obj[attr].get('ub', None)
                attrs_dict[a].append(ub)

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
            cur.execute("SELECT DISTINCT({0}) FROM ")
    cur.execute(q)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    print(res_df)
    break;
cur.close()
conn.close()
