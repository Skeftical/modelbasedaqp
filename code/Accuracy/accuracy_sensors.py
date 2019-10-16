import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from scipy import stats

from sklearn.cluster import KMeans

#Metrics
from sklearn import metrics
import sys
import os
os.chdir('../../')
sys.path.append('.')
from confs import Config
from terminal_outputs import printProgressBar

def load_data():
    print("Loading Data...")
    global df
    df = pd.read_csv('input/Sensors_Workload/queries_on_c_all_aggregates-{}.csv'.format(Config.sensors_queries), sep=",", index_col=0)
    df = df.drop(['corr','avg','count','sum_'], axis=1)
    df['x_l'] = df['x']-df['theta']
    df['x_h'] = df['x']+df['theta']
    df = df.drop(['x','theta'],axis=1)


def run_experiment():
    no_queries = np.linspace(0.1,1,10)*(15000)
    agg_label = ['min_', 'max_']
    labels = ['MIN','MAX']
    alter_columns_1 = ['x_l', 'x_h']
    t_cuttoff = int(Config.sensors_queries*0.8)
    rel_errs_ml_queries = []

    for l1,s in zip(agg_label,labels):

        for no in no_queries:
            no = int(no)
            test = df.sample(int(no*.2))
            train = df.drop( test.index,axis='index').sample(no)
            X_test = test[alter_columns_1].values
            y_test = test[l1].values
            X_train = train[alter_columns_1].values
            y_train = train[l1].values

            print(" Zeros {}".format((y_test==0).sum()/y_test.shape[0]))
            if (y_test==0).any():
                print("Shape was {}".format(y_test.shape[0]))
                X_test = X_test[y_test!=0]
                y_test = y_test[y_test!=0]
            print("Reconfigured shape {}".format(y_test.shape[0]))

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            lgb = LGBMRegressor()

            lgb.fit(X_train, y_train)


            y_pred = lgb.predict(X_test)
            y_pred = y_pred.reshape(y_test.shape[0],)
            print(stats.describe(y_pred))
            print(stats.describe(y_test))

            rel_error_ML_sum = np.mean(np.abs(y_pred-y_test)/y_test)
            nrmsd = np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            mae = metrics.median_absolute_error(y_test, y_pred)
            rel_errs_ml_queries.append([no, s, rmse, mae, nrmsd, rel_error_ML_sum])

    eval_df = pd.DataFrame(rel_errs_ml_queries, columns=['queries', 'aggregate','rmse','mae','nrmsd', 'rel_error_median'])
    eval_df.to_csv('output/accuracy/csvs/sensors_assessment_on_queries_{}_queries.csv'.format(Config.sensors_queries))

if __name__=='__main__':
    np.random.seed(0)
    load_data()
    run_experiment()
