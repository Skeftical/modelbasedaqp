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
from sklearn.model_selection import train_test_split



if not os.path.exists('output/performance'):
        print('creating ', 'performance')
        os.makedirs('output/performance')
if not os.path.exists('output/performance/csvs'):
        print('creating ', 'performance csvs')
        os.makedirs('output/performance/csvs')


qdf = pd.read_pickle('input/instacart_queries/qdf.pkl')
targets = [name  for name in qdf.columns if 'lb' not in name and 'ub' not in name]
#Filtering out the product of joins in the aggregate functions
count_columns = [name for name in qdf[targets].columns if 'count' in name]
#Generate Dataframes per aggregate function
count_df = qdf.iloc[qdf['count'].dropna(axis=0).index]

features = [name for name in count_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]


target_count = 'count'
count_df = count_df[(count_df[target_count]!=0)] # Remove 0 because it produces an error on relative error
count_df['product_name_lb'] = count_df['product_name_lb'].replace(np.nan, 'isnone')
count_df['product_name_lb'] = count_df['product_name_lb'].astype('category')
labels =  count_df['product_name_lb'].cat.codes
categorical_attribute_catalogue = {key : value for key,value in zip(count_df['product_name_lb'].values, labels)}
count_df['product_name_lb'] = labels

del qdf
no_queries = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
query_results = {}
query_results['no_queries'] = []
query_results['time'] = []
# # read in data
for no in no_queries:

    X = count_df.loc[:no, features].values
    y = count_df.loc[:no, target_count].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    dtrain = xgb.DMatrix(X_train,y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, y_test, feature_names=features)

    params = {'max_depth':6, 'eta':0.3, 'objective': 'reg:squarederror', 'eval_metric':['rmse'],'colsample_bytree':0.75, 'colsample_bylevel':0.75, 'colsample_bynode':0.75, 'reg_alpha':0.3, 'reg_lambda':1}
    for i in range(10):
        start = time.time()
        xgb_model = xgb.train(params, dtrain,num_boost_round=1000,early_stopping_rounds=10, evals=[(dtrain,'train'),(dtest,'test')],
             verbose_eval=True)
        end = time.time()-start
        query_results['no_queries'].append(no)
        query_results['time'].append(end)
    print("Time to train for {} \t took : {}+-".format(no, np.mean(query_results['time']), np.std(query_results['time'])))

    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
df = pd.DataFrame(query_results)
df.to_csv('output/performance/csvs/query_training.csv')
