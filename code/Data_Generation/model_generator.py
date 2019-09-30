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
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

MODEL_CATALOGUE = {}


qdf = pd.read_pickle('input/instacart_queries/qdf.pkl')
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
avg_df[target_avg] = avg_df[target_avg].astype(float)
models_train = [(sum_df, target_sum, 'sum_add_to_cart_order'), (avg_df, target_avg, 'avg_add_to_cart_order'), (count_df, target_count, 'count')]

del qdf
# # read in data
for df, label,af in models_train:

    if label=='count':
        df = df.groupby('product_name_lb', group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))
        print("Resulting sample {}".format(df.shape))
        df['product_name_lb'] = df['product_name_lb'].astype(str)
        X = df[features].values
        y = df[label].values
        y = y.astype(int)
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

        cat_model=CatBoostRegressor(iterations=1000, depth=3, learning_rate=0.1, loss_function='RMSE')
        cat_model.fit(X_train, y_train,cat_features=[12],eval_set=(X_validation, y_validation))

        rel_error = relative_error(y_validation, cat_model.predict(X_validation))
        ml_est = MLAF(cat_model, rel_error, features, label)
        MODEL_CATALOGUE[af] = ml_est
        print("Relative Error for {} is {}".format(label, rel_error))
        continue;

    X = df[features].values
    y = df[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    dtrain = xgb.DMatrix(X_train,y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    params = {'max_depth':5, 'eta':0.3, 'objective':'reg:squarederror', 'eval_metric':['rmse'],'colsample_bytree':0.75, 'colsample_bylevel':0.75, 'colsample_bynode':0.75, 'reg_alpha':0.3, 'reg_lambda':1}
    start = time.time()

    xgb_model = xgb.train(params, dtrain,num_boost_round=1000,early_stopping_rounds=10, evals=[(dtrain,'train'),(dtest,'test')],
         verbose_eval=True)

    rel_error =relative_error(y_test, xgb_model.predict(dtest))
    print("Relative Error for {} is {}".format(label, rel_error))
    print("Time to train for {} \t took : {}".format(label, time.time()-start))

    ml_est = MLAF(xgb_model, rel_error, features, label)
    MODEL_CATALOGUE[af] = ml_est
    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
with open('catalogues/model_catalogue.pkl', 'wb') as f:
    pickle.dump(MODEL_CATALOGUE, f)
