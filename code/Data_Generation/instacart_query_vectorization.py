import os
import sys
os.chdir('../../')
import pickle
import psycopg2
import pandas as pd

queries = []
with open('input/instacart_queries/queries.pkl','rb') as f:
    queries = pickle.load(f)

conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='instacart',user='analyst',password='analyst')
cur = conn.cursor()

for q in queries:
    print(q)
    cur.execute(q)
    res = cur.fetchall()
    res_df = pd.DataFrame(res)

    print(res_df)
    break;
cur.close()
conn.close()
