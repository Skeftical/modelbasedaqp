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
from sklearn.metrics import r2_score

def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

def f_relative_error(y_true: np.ndarray, dtrain: xgb.DMatrix):
    y_hat = dtrain.get_label()
    return 'RelativeError' , float(np.mean(np.abs((y_true-y_hat)/y_true)))

MODEL_CATALOGUE = {}


qdf = pd.read_pickle('input/instacart_queries/qdf.pkl')
#Transformations

qdf['group'] = ((~qdf['product_name_lb'].isna()) | (~qdf['reordered_lb'].isna()) | (~qdf['order_hour_of_day_lb'].isna()))
qdf['group_by_attr'] = [False for i in range(qdf.shape[0])]

qdf.loc[~qdf['reordered_lb'].isna(),'group_by_attr'] = 'group_by_reordered'
qdf.loc[~qdf['product_name_lb'].isna(),'group_by_attr'] = 'group_by_product'
qdf.loc[~qdf['order_hour_of_day_lb'].isna(),'group_by_attr'] = 'group_by_order_hour'


#Generate Dataframes per aggregate function
sum_df = qdf.iloc[qdf['sum_add_to_cart_order'].dropna(axis=0).index]
avg_df = qdf.iloc[qdf['avg_add_to_cart_order'].dropna(axis=0).index]
count_df = qdf.iloc[qdf['count'].dropna(axis=0).index]


target_sum = 'sum_add_to_cart_order'
target_avg = 'avg_add_to_cart_order'
target_count = 'count'
count_df = count_df[(count_df[target_count]!=0)] # Remove 0 because it produces an error on relative error
count_df['product_name_lb'] = count_df['product_name_lb'].replace(np.nan, 'isnone')
count_df['product_name_lb'] = count_df['product_name_lb'].astype('category')
labels =  count_df['product_name_lb'].cat.codes
categorical_attribute_catalogue = {key : value for key,value in zip(count_df['product_name_lb'].values, labels)}
count_df['product_name_lb'] = labels

with open('catalogues/labels_catalogue.pkl', 'wb') as f:
    pickle.dump(categorical_attribute_catalogue,f)

avg_df[target_avg] = avg_df[target_avg].astype(float)
sum_df[target_sum] = sum_df[target_sum].astype(float)
count_df[target_count] = count_df[target_count].astype(float)

del qdf

sum_df = pd.get_dummies(sum_df)
avg_df = pd.get_dummies(avg_df)
count_df = pd.get_dummies(count_df)

features_sum = [name for name in sum_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]
features_avg = [name for name in avg_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]
features_count = [name for name in count_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]

models_train = [(sum_df, target_sum, 'sum_add_to_cart_order', features_sum), \
(avg_df, target_avg, 'avg_add_to_cart_order', features_avg), (count_df, target_count, 'count', features_count)]

# # read in data
for df, label,af,features in models_train:
    print(df)
    X = df[features].values
    y = df[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    dtrain = xgb.DMatrix(X_train,label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    print((dtrain.num_row(), dtrain.num_col()))
    print((dtest.num_row(), dtest.num_col()))
    params = {'max_depth':10,'tree_method':'exact', 'eta':0.1, 'objective': 'reg:squarederror', 'reg_alpha':0.3, 'reg_lambda':0.75, 'eval_metric': ['mae','rmse']}
    start = time.time()
    xgb_model = xgb.train(params, dtrain, num_boost_round=2000, early_stopping_rounds=10, feval=f_relative_error, evals=[(dtrain,'train'),(dtest,'test')])
    rel_error =relative_error(y_test, xgb_model.predict(dtest))
    print("Relative Error for {} is {}".format(label, rel_error))
    print("Time to train for {} \t took : {}".format(label, time.time()-start))

    ml_est = MLAF(xgb_model, rel_error, features, label)
    MODEL_CATALOGUE[af] = ml_est
    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
with open('catalogues/model_catalogue.pkl', 'wb') as f:
    pickle.dump(MODEL_CATALOGUE, f)
