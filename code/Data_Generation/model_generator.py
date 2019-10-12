import os
import sys
os.chdir('../../')
#print(os.listdir('.'))
sys.path.append('.')
import pickle
import pandas as pd
import numpy as np
from ml.model import MLAF
from xgboost import XGBRegressor
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

#def f_relative_error(y_true: np.ndarray, dtrain: xgb.DMatrix):
#    y_hat = dtrain.get_label()
#    return 'RelativeError' , float(np.mean(np.abs((y_true-y_hat)/y_true)))

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
count_df['product_name_lb'] = count_df['product_name_lb'].replace(np.nan, 'isnone')
count_df['product_name_lb'] = count_df['product_name_lb'].astype('category')
labels =  count_df['product_name_lb'].cat.codes
categorical_attribute_catalogue = {key : value for key,value in zip(count_df['product_name_lb'].values, labels)}
count_df['product_name_lb'] = labels

with open('catalogues/labels_catalogue.pkl', 'wb') as f:
    pickle.dump(categorical_attribute_catalogue,f)

avg_df[target_avg] = avg_df[target_avg].astype(float)
models_train = [(sum_df, target_sum, 'sum_add_to_cart_order'), (avg_df, target_avg, 'avg_add_to_cart_order'), (count_df, target_count, 'count')]

del qdf
# # read in datai
rel_error_score = make_scorer(relative_error, greater_is_better=False)
for df, label,af in models_train:

    X = df[features].values
    y = df[label].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    # dtrain = xgb.DMatrix(X_train,label=y_train, feature_names=features)
    # dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    # params = {'max_depth':6, 'eta':0.2,, 'reg_alpha':0.3, 'reg_lambda':1, 'eval_metric': ['rmse', r2_score, f_relative_error]}
    start = time.time()
    # xgb_model = xgb.train(params, dtrain,, num_boost_round=1000, early_stopping_rounds=10, feval=f_relative_error, evals=[(dtrain,'train'),(dtest,'test')])
    params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [3,4, 6]}

    # Initialize XGB and GridSearch
    xgb = XGBRegressor(nthread=-1, n_estimators=500)

    grid = GridSearchCV(xgb, params,n_jobs=-1, scoring=rel_error_score)
    grid.fit(X, y)
    print("Best Parameters : {}".format(grid.best_params_))
    print("Best Score : {}".format(grid.best_score_))
    xgb_model = grid.best_estimator_

    rel_error =relative_error(y, xgb_model.predict(X))
    print("Relative Error for {} is {}".format(label, rel_error))
    print("Time to train for {} \t took : {}".format(label, time.time()-start))

    ml_est = MLAF(xgb_model, rel_error, features, label)
    MODEL_CATALOGUE[af] = ml_est
    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
with open('catalogues/model_catalogue.pkl', 'wb') as f:
    pickle.dump(MODEL_CATALOGUE, f)
