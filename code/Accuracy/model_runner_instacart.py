import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import psycopg2
import pandas as pd
from sql_parser.parser import Parser
import numpy as np
import time

if not os.path.exists('output/model-based/instacart'):
        os.makedirs('output/model-based/instacart')

with open('input/instacart_queries/queries-test.pkl', 'rb') as f:
    queries = pickle.load(f)

with open('catalogues/distinct_attribute_catalogue.pkl', 'rb') as f:
    distinct_attr = pickle.load(f)

with open('catalogues/model_catalogue.pkl', 'rb') as f:
    model_catalogue = pickle.load(f)

with open('catalogues/labels_catalogue.pkl', 'rb') as f:
    labels_catalogue = pickle.load(f)

print(model_catalogue)
query_answers_dic = {}
query_answers_dic['query_name'] = []
query_answers_dic['time'] = []
query_names = {}
i = 0
for qname,q in queries:
    print("Query {}".format(q))
    print(q)
    pr = Parser()
    pr.parse(q)
    dict_obj = pr.get_vector()
    proj_dict = pr.get_projections()
    print(proj_dict)
    gattr = pr.get_groupby_attrs()
    print(gattr)
    print(dict_obj)
    #####
    # Estimation Phase
    res = {}
    start = time.time()
    for p in proj_dict:
        res[p] = []
        est = model_catalogue[p]
        if len(gattr)>0:
            for g in gattr:
                gvalues = distinct_attr[g]
                print("length of groupby values {}".format(len(gvalues)))
                res[g] = []
                for gval in gvalues:
                    dict_obj[g+'_lb'] = labels_catalogue.get(gval,np.nan)
                    res[p].append(est.predict_one(dict_obj))
                    res[g].append(gval)

        else:
            res[p].append(est.predict_one(dict_obj))
    end = time.time()-start
    res_df = pd.DataFrame(res)
    print(res_df.describe())
    res_df.to_pickle('output/model-based/instacart/{}.pkl'.format(i))

    #####
    query_answers_dic['time'].append(end)
    query_answers_dic['query_name'].append(qname)
    if qname not in query_names:
        query_names[qname] = [i]
    else:
        query_names[qname].append(i)
    i+=1
    print("{}/{} Queries Processed ================".format(i,len(queries)))

qa = pd.DataFrame(query_answers_dic)
qa.to_csv('output/model-based/instacart/query-response-time.csv')
with open('output/model-based/instacart/query-assoc-names.pkl', 'wb') as f:
    pickle.dump(query_names, f)
