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
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public';")
res = cur.fetchall()
set_of_attributes = set(map(lambda x: x[0].replace('_',''),res))
attrs_array = []
for a in set_of_attributes:
	attrs_array.append('_'.join([a,'lb']))
	attrs_array.append('_'.join([a,'ub']))
print(attrs_array)
attrs_dict = { key : [] for key in attrs_array } #dict.fromkeys(attrs_array,[[]]*len(attrs_array))
afs = {}

for q in queries:
    print(q)
    pr.parse(q)
    cur.execute(q)


    dict_obj = pr.get_vector()
    print(dict_obj)
    for a in attrs_dict:
        if a=='af':
             continue;
        attr, dire = a.split('_')
        print((attr,dire))
        if attr not in dict_obj:#If this attribute is not in the query vector then leave as None
            attrs_dict[a].append(None) # Fill with None initially
        else:
            if dire=='lb':
                lb = dict_obj[attr].get('lb',None)
                attrs_dict[a].append(lb)
            else:
                ub = dict_obj[attr].get('ub', None)
                attrs_dict[a].append(ub)
    print(attrs_dict)

    proj_dict = pr.get_projections()
    print(proj_dict)

    print(pr.get_groupby_attrs())
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    print(res_df)
    break;
cur.close()
conn.close()
