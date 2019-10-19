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
from sklearn.metrics import r2_score

def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

def f_relative_error(y_true: np.ndarray, dtrain: lgb.Dataset):
    y_hat = dtrain.get_label()
    return ('RelativeError' , float(np.mean(np.abs((y_true-y_hat)/y_true))), False)

MODEL_CATALOGUE = {}


qdf = pd.read_pickle('input/instacart_queries/qdf-1000.pkl')
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
count_df['product_name_lb'] = labels
categorical_attribute_catalogue = {key : value for key,value in zip(count_df['product_name_lb'].values, labels)}
# count_df['product_name_lb'] = labels

with open('catalogues/labels_catalogue.pkl', 'wb') as f:
    pickle.dump(categorical_attribute_catalogue,f)

avg_df[target_avg] = avg_df[target_avg].astype(float)
sum_df[target_sum] = sum_df[target_sum].astype(float)
count_df[target_count] = count_df[target_count].astype(float)

avg_df = avg_df.infer_objects()
sum_df = sum_df.infer_objects()
count_df = count_df.infer_objects()

models_train = [(sum_df, target_sum, 'sum_add_to_cart_order'), (avg_df, target_avg, 'avg_add_to_cart_order'), (count_df, target_count, 'count')]

del qdf
# # read in data
for df, label,af in models_train:
    print("For Aggregate {}".format(af))
    X = df[features]
    y = df[label]
    print(df[features].dtypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    dtrain = lgb.Dataset(X_train,label=y_train )
    dtest = lgb.Dataset(X_test, label=y_test)

    param = {'num_leaves': 50,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}
    start = time.time()
    num_round = 15000
    clf = lgb.train(param, dtrain, num_round, valid_sets = [dtrain,dtest],valid_names=['train', 'test'],\
    feval=f_relative_error,verbose_eval=100, early_stopping_rounds=100)
    rel_error =relative_error(y_test, clf.predict(X_test))
    print("Relative Error for {} is {}".format(label, rel_error))
    print("Time to train for {} \t took : {}".format(label, time.time()-start))

    ml_est = MLAF(clf, rel_error, features, label)
    MODEL_CATALOGUE[af] = ml_est
    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
with open('catalogues/model_catalogue.pkl', 'wb') as f:
    pickle.dump(MODEL_CATALOGUE, f)
