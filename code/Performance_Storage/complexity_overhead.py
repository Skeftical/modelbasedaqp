import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import pandas as pd
import numpy as np
from ml.model import MLAF
import xgboost as xgb
import time



if not os.path.exists('output/performance'):
        print('creating ', 'performance')
        os.makedirs('output/performance')
if not os.path.exists('output/performance/csvs'):
        print('creating ', 'performance csvs')
        os.makedirs('output/performance/csvs')
if not os.path.exists('output/performance/complexity_store'):
        print('creating ', 'performance complexity_store')
        os.makedirs('output/performance/complexity_store')


qdf = pd.read_pickle('input/instacart_queries/qdf.pkl')
targets = [name  for name in qdf.columns if 'lb' not in name and 'ub' not in name]
#Filtering out the product of joins in the aggregate functions
sum_columns = [name for name in qdf[targets].columns if 'sum' in name]
#Generate Dataframes per aggregate function
sum_df = qdf.iloc[qdf['sum_add_to_cart_order'].dropna(axis=0).index]

features = [name for name in sum_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]


target_sum = 'sum_add_to_cart_order'


del qdf
query_results = {}
query_results['boosting'] = []
query_results['max_depth'] = []
query_results['storage'] = []
query_results['time'] = []
query_results['size'] = []
boosting = [10, 100, 300, 500, 1000]
max_depth = [1,3,5,7,9,12]
# # read in data
for b in boosting:
    for d in max_depth

        X = sum_df.loc[:, features].values
        y = sum_df.loc[:, label].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
        dtrain = xgb.DMatrix(X_train,y_train, feature_names=features)
        dtest = xgb.DMatrix(X_test, y_test, feature_names=features)

        params = {'max_depth':d, 'eta':0.3, 'objective':"reg:squarederror", 'eval_metric':['rmse'],'colsample_bytree':0.75, 'colsample_bylevel':0.75, 'colsample_bynode':0.75, 'reg_alpha':0.3, 'reg_lambda':1}
        for i in range(5):
            start = time.time()
            xgb_model = xgb.train(params, dtrain,num_boost_round=b, evals=[(dtrain,'train'),(dtest,'test')],
                 verbose_eval=True)
            end = time.time()-start
            query_results['boosting'].append(b)
            query_results['max_depth'].append(d)
            query_results['time'].append(end)
        pkl_filename = "pickle_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(m, file)
        statinfo = os.stat('pickle_model.pkl')
        query_results['size'].append(statinfo.st_size)
        print("Time to train for {} \t took : {}+-".format((b,d), np.mean(query_results['time']), np.std(query_results['time'])))

    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
df = pd.DataFrame(query_results)
df.to_csv('output/performance/csvs/complexity_overhead.csv')
