import os
import sys
os.chdir('../../')
import pickle
import psycopg2
import pandas as pd
from sql_parser.parser import Parser

queries = []
with open('input/instacart_queries/queries.pkl','rb') as f:
    queries = pickle.load(f)
pr = Parser()
conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='instacart',user='analyst',password='analyst')
cur = conn.cursor()
#Obtain all attribute columns
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public';")
res = cur.fetchall()
print(res)

for q in queries:
    print(q)
    pr.parse(q)
    cur.execute(q)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)
    print(pr.get_vector())
    print(pr.get_projections())
    print(pr.get_groupby_attrs())
    print(res_df)
    break;
cur.close()
conn.close()
