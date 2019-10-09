import json
import requests
import os
import sys
os.chdir('../../')
print(os.listdir('.'))
sys.path.append('.')
import pickle
import psycopg2
import pandas as pd
from sql_parser.parser import Parser
import numpy as np
import time

if not os.path.exists('output/model-based-networn/instacart'):
        os.makedirs('output/model-based-network/instacart')


query_answers_dic = {}
query_answers_dic['query_name'] = []
query_answers_dic['time'] = []
i = 0
for qname,q in queries:
    print("Query {}".format(q))
    print(q)
    pr = Parser()
    pr.parse(q)
    event = {}
    event['projections'] = pr.get_projections()
    event['groups'] = pr.get_groupby_attrs()
    event['filters']= pr.get_vector()
    #####
    # Estimation Phase
    start = time.time()
    res = requests.post("http://localhost:5000", json=event)
    end = time.time()-start
    #####
    query_answers_dic['time'].append(end)
    query_answers_dic['query_name'].append(qname)

    i+=1
    print("{}/{} Queries Processed ================".format(i,len(queries)))

qa = pd.DataFrame(query_answers_dic)
qa.to_csv('output/model-based/instacart/query-response-time.csv')
