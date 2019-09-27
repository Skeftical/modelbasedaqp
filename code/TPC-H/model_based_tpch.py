import pandas as pd
import numpy as np
#ML Preprocessing
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
#Metrics
from sklearn import metrics
import pickle
import os
import time

def relative_error(y_true, y_hat):
    return np.mean(np.abs((y_true-y_hat)/y_true))

if not os.path.exists('../../output/model-based/tpch'):
        # logger.info('creating directory Accuracy')
        os.makedirs('../../output/model-based/tpch')

df1 = pd.read_csv('../../input/tpch_queries/tpch-queries-1.csv', header=0, index_col=0)
df3 = pd.read_csv('../../input/tpch_queries/tpch-queries-3-v2.csv', header=0, index_col=0)
df4 = pd.read_csv('../../input/tpch_queries/tpch-queries-4-v2.csv', header=0, index_col=0)
df5 = pd.read_csv('../../input/tpch_queries/tpch-queries-5-v2.csv', header=0, index_col=0)
df6 = pd.read_csv('../../input/tpch_queries/tpch-queries-6.csv', header=0, index_col=0)

tpch_queries = ['tpch1', 'tpch3', 'tpch4', 'tpch5', 'tpch6']
tpch_dfs = [df1, df3, df4, df5, df6]

features1 = ['interval', 'l_returnflag ', 'l_linestatus ']
target1 = ['sum_qty  ', 'sum_base_price  ', 'sum_disc_price   ', 'sum_charge      ', 'avg_qty       ', 'avg_price      ', 'avg_disc        ', 'count_order ']
features3 = [c for c in df3.columns if c!='revenue   ']
target3 = ['revenue   ']
target4 = ['order_count ']
features4 = [c for c in df4.columns if c!='order_count ']
target5 = ['revenue    ']
features5 = [c for c in df5.columns if c!='revenue    ']
target6 = ['revenue']
features6 = [c for c in df6.columns if c!='revenue']

test_indices = np.random.randint(0,100,20)
k=0
for i,t in zip([1,3,4,5,6],tpch_dfs):
    tpch_dfs[k] = t.set_index(map(lambda x: str(i)+'-'+str(x),t.index))
    k+=1

complete_df = pd.concat(tpch_dfs, sort=False)

tpch_features = features1 + features3 + features4 + features5 + features6
tpch_targets = target1 + target3 + target4 + target5 + target6

training_indices = set([ k for k in filter(lambda x : ~(int(x.split('-')[1])==test_indices).any(), complete_df.index)])

complete_df_test = complete_df.drop(index=complete_df.loc[training_indices].index)
complete_df = complete_df.loc[training_indices]

estimators = {}
for target in tpch_targets:
    print("Target :{}".format(target))
    temp = complete_df.loc[complete_df[target].dropna(axis=0).index]
    X = temp[tpch_features].values
    y = temp[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    estimators[target] = xgb_model
    print(relative_error(y_test, xgb_model.predict(X_test)))

query_answers_dic = {}
query_answers_dic['query_name'] = []
query_answers_dic['time'] = []
avg_rel_error_queries = {}
for ix in complete_df_test.index:
    q = complete_df_test.loc[[ix]]
    print("Query : {}".format(ix))
    X = q[tpch_features].values
    targets = q[tpch_targets].dropna(axis=1).columns
    rel_errors_per_aggregate = []
    start = time.time()
    for t in targets:
        print(t)
        y = q[t].values
        rel_error = relative_error(y, estimators[t].predict(X))
        print("Relative Error for {} at query {} : {}".format(t,ix,rel_error))
        rel_errors_per_aggregate.append(rel_error)
    avg_rel = np.mean(rel_errors_per_aggregate)
    print("Average relative error at query {} is : {}".format(ix, avg_rel))
    end = time.time()-start
    query_answers_dic['time'].append(end)
    query_answers_dic['query_name'].append(ix.split('-')[0])
    if ix.split('-')[0] in avg_rel_error_queries:
        avg_rel_error_queries[ix.split('-')[0]].append(avg_rel)
    else:
        avg_rel_error_queries[ix.split('-')[0]] = [avg_rel]

qa = pd.DataFrame(query_answers_dic)
qa.to_csv('../../output/model-based/tpch/query-response-time.csv')
with open('../../output/model-based/tpch/query-errors', 'wb') as f:
    pickle.dump(avg_rel_error_queries, f)
