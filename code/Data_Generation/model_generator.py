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


def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

MODEL_CATALOGUE = {}


qdf = pd.read_pickle('input/instacart_queries/qdf.pkl')
targets = [name  for name in qdf.columns if 'lb' not in name and 'ub' not in name]
#Filtering out the product of joins in the aggregate functions
sum_columns = [name for name in qdf[targets].columns if 'sum' in name]
avg_columns = [name for name in qdf[targets].columns if 'avg' in name]
count_columns = [name for name in qdf[targets].columns if 'count' in name]
# #Dropping unnecessary columns and colapsing everything into one column
# qdf['sum_af'] = qdf[sum_columns].apply(lambda x: float(x.dropna()) if not x.dropna().empty else np.nan,axis=1)
# qdf = qdf.drop(columns=sum_columns)
# qdf['avg_af'] = qdf[avg_columns].apply(lambda x: float(x.dropna()) if not x.dropna().empty else np.nan,axis=1)
# qdf = qdf.drop(columns=avg_columns)
# qdf['count_af'] = qdf[count_columns].apply(lambda x: float(x.dropna()) if not x.dropna().empty else np.nan,axis=1)
# qdf = qdf.drop(columns=count_columns)
#Generate Dataframes per aggregate function
sum_df = qdf.iloc[qdf['sum_add_to_cart_order'].dropna(axis=0).index]
avg_df = qdf.iloc[qdf['avg_add_to_cart_order'].dropna(axis=0).index]
count_df = qdf.iloc[qdf['count'].dropna(axis=0).index]

features = [name for name in sum_df.columns if name not in ['sum_add_to_cart_order','avg_add_to_cart_order','count']]

target_sum = 'sum_add_to_cart_order'
target_avg = 'avg_add_to_cart_order'
target_count = 'count'
count_df = count_df[(count_df[target_count]!=0)] # Remove 0 because it produces an error on relative error

models_train = [(sum_df, target_sum, 'sum_add_to_cart_order'), (avg_df, target_avg, 'avg_add_to_cart_order'), (count_df, target_count, 'count')]
# # read in data
for df, label,af in models_train:
    if label=='count':
        labels =  df['product_name_lb'].astype('category').cat.codes
        categorical_attribute_catalogue = {key : value for key,value in zip(df['product_name_lb'].values, labels)}
        df['product_name_lb'] = labels
        with open('catalogues/labels_catalogue.pkl', 'wb') as f:
            pickle.dump(categorical_attribute_catalogue,f)
#         obj = 'count:poisson'
    X = df[features].values
    y = df[label].values.astype(float)
    dtrain = xgb.DMatrix(X,y)
    # dtest = xgb.DMatrix(y)
    # # specify parameters via map
    param = {'max_depth':3, 'eta':1, 'objective':'reg:squarederror'}
    num_round = 20
    # bst = xgb.train(param, dtrain, num_round)
    xgb_model = xgb.train(param, dtrain, num_round)
    rel_error =relative_error(y, xgb_model.predict(dtrain))
    print("Relative Error for {} is {}".format(label, rel_error))
    ml_est = MLAF(xgb_model, rel_error, features)
    MODEL_CATALOGUE[af] = ml_est
    # xgb_model.save_model('/home/fotis/dev_projects/model-based-aqp/catalogues/{}.dict_model'.format(label))
with open('catalogues/model_catalogue.pkl', 'wb') as f:
    pickle.dump(MODEL_CATALOGUE, f)
