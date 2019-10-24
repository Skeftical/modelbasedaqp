import pandas as pd
import numpy as np
#Metrics
from sklearn import metrics
import pickle
import lightgbm as lgb
import time
import os
import sys
from sklearn.model_selection import train_test_split
os.chdir('../../')
print(os.listdir('.'))
sys.path.append('.')
def relative_error(y_true, y_hat):
    return float(np.mean(np.abs((y_true-y_hat)/y_true)))

def f_relative_error(y_true: np.ndarray, dtrain: lgb.Dataset):
    y_hat = dtrain.get_label()
    return ('RelativeError' , np.mean(np.abs((y_true-y_hat)/y_true)), False)

qdf = pd.read_pickle('/home/fotis/dev_projects/model-based-aqp/input/instacart_queries/qdf-1000.pkl')

targets = [name  for name in qdf.columns if 'lb' not in name and 'ub' not in name]
#Filtering out the product of joins in the aggregate functions
sum_columns = [name for name in qdf[targets].columns if 'sum' in name]
avg_columns = [name for name in qdf[targets].columns if 'avg' in name]
count_columns = [name for name in qdf[targets].columns if 'count' in name]
#Generate Dataframes per aggregate function
sum_df = qdf.iloc[qdf['sum_add_to_cart_order'].dropna(axis=0).index]
avg_df = qdf.iloc[qdf['avg_add_to_cart_order'].dropna(axis=0).index]
count_df = qdf.iloc[qdf['count'].dropna(axis=0).index]

features = [name for name in sum_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]

target_sum = 'sum_add_to_cart_order'
target_avg = 'avg_add_to_cart_order'
target_count = 'count'
count_df = count_df[(count_df[target_count]!=0)] # Remove 0 because it produces an error on relative error
# count_df['product_name_lb'] = count_df['product_name_lb'].replace(np.nan, 'isnone')
count_df['product_name_lb'] = count_df['product_name_lb'].astype('category')
labels =  count_df['product_name_lb'].cat.codes
categorical_attribute_catalogue = {key : value for key,value in zip(count_df['product_name_lb'].values, labels)}
count_df['product_name_lb'] = labels


avg_df[target_avg] = avg_df[target_avg].astype(float)
sum_df[target_sum] = sum_df[target_sum].astype(float)
count_df[target_count] = count_df[target_count].astype(float)

avg_df = avg_df.infer_objects()
sum_df = sum_df.infer_objects()
count_df = count_df.infer_objects()

count_df_s = count_df.sample(frac=.25)

models_train = [(sum_df, target_sum, 'sum_add_to_cart_order'), (avg_df, target_avg, 'avg_add_to_cart_order'), (count_df_s, target_count, 'count')]

del qdf

results = {}
results['count'] = []
results['avg_add_to_cart_order'] = []
results['sum_add_to_cart_order'] = []

ALPHA=.95

for df, label,af in models_train:
    print("For Aggregate {}".format(af))
    X = df[features]
    y = df[label]
    print(df[features].dtypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

    start = time.time()
    y_mean = y_test

    for i in range(5):
        print("Run {}/5".format(i+1))
        print((X_train.shape, y_train.shape))
        clf_upper = lgb.LGBMRegressor(n_estimators=1500,objective='quantile',alpha=ALPHA, learning_rate=0.001)
        clf_upper.fit(X_train, y_train)
        y_upper = clf_upper.predict(X_test)
        
        clf_lower = lgb.LGBMRegressor(n_estimators=1500,objective='quantile',alpha=1.-ALPHA, learning_rate=0.001)
        clf_lower.fit(X_train, y_train)
        y_lower = clf_lower.predict(X_test)

        error_est_df = pd.DataFrame(np.column_stack((y_mean, y_lower, y_upper)),columns=['y_', 'y_l', 'y_u'])
        coverage = (error_est_df.apply(lambda x: x['y_l']<=x['y_']<=x['y_u'],axis=1).value_counts()/error_est_df.shape[0])[True]
        width = np.mean(error_est_df['y_u']-error_est_df['y_l'])
        results[label].append((coverage, width))
     
df = pd.DataFrame(results)
df.to_csv('output/model-based/error_estimation/results.csv')
