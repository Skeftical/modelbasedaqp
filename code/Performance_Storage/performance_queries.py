import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import pandas as pd
import numpy as np
from ml.model import MLAF
import lightgbm as lgb
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

X = count_df[features].values
y = count_df['count'].values
model = lgb.LGBMRegressor(n_estimators=5000,)
queries_no = np.linspace(10000, X.shape[0],20,dtype=int)
query_results = {}
query_results['no_queries'] = []
query_results['time'] = []
query_results['columns'] = []
cols = [1,2,3,4,5]

# # read in data
for col in cols:
    for no in queries_no:

        if col>1:
            X = np.hstack(tuple([X for i in range(col)]))

        time_runs = []

        print("Running for {} queries".format(no))
        for _ in range(10):
            start = time.time()
            model.fit(X[:no],y[:no])
            end = time.time()-start
            query_results['no_queries'].append(no)
            query_results['time'].append(end)
            query_results['columns'].append(X.shape[1])
        print("Time to train for {} \t took : {}+-".format(no, np.mean(query_results['time']), np.std(query_results['time'])))

    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
df = pd.DataFrame(query_results)
df.to_csv('output/performance/csvs/query_training.csv')
