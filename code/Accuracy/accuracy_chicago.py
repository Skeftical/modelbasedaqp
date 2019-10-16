import pandas as pd
import numpy as np
from scipy import stats
#ML Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Models
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
#Metrics
from sklearn import metrics
import sys
import os
os.chdir('../../')
sys.path.append('.')
from confs import Config
from ML_models.GrowingMiniBatchKMeans import GrowingMiniBatchKMeans
from ML_models.GrowingNetwork import GrowingNetwork
from ML_models.LocalSupervisedNetwork import LocalSupervisedNetwork, LocalSupervisedOfflineNetwork
from terminal_outputs import printProgressBar


def load_data():
    print("Loading Data...")
    global train_df
    global test_df

    train_df = pd.read_csv('input/Crimes_Workload/train_workload_gauss-{}-users-{}.csv'.format(Config.chicago_clusters,Config.chicago_queries), header=0, index_col=0)
    test_df = pd.read_csv('input/Crimes_Workload/test_workload_gauss-{}-users-{}.csv'.format(Config.chicago_clusters,Config.chicago_queries), header=0, index_col=0)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df.dropna()
    train_df = train_df.dropna()
    test_df['count'] = test_df['count'].apply(np.exp)
    test_df['sum_'] = test_df['sum_'].apply(np.exp)
    train_df['count'] = train_df['count'].apply(np.exp)
    train_df['sum_'] = train_df['sum_'].apply(np.exp)

    test_df['count'] = test_df['count'].replace([np.inf, -np.inf], np.nan).dropna()
    test_df['sum_'] = test_df['sum_'].replace([np.inf, -np.inf], np.nan).dropna()
    train_df['count'] = train_df['count'].replace([np.inf, -np.inf], np.nan).dropna()
    train_df['sum_'] = train_df['sum_'].replace([np.inf, -np.inf], np.nan).dropna()
    cutoff = train_df.count()[0]
    complete = pd.concat([train_df, test_df], ignore_index=True)
    complete['x_l'] = complete['x']-complete['x_range']
    complete['x_h'] = complete['x']+complete['x_range']
    complete['y_l'] = complete['y']-complete['y_range']
    complete['y_h'] = complete['y']+complete['y_range']
    complete = complete.drop(['x','y','x_range','y_range'],axis=1)
    train_df = complete.iloc[:cutoff]
    test_df = complete.iloc[cutoff:]


def assess_on_queries():
    no_queries = np.linspace(0.1,1,10)*(25000)

    agg_label = ['count', 'sum_', 'avg']
    labels = ['COUNT','SUM','MEAN']
    alter_columns_1 = ['x_l', 'x_h', 'y_l', 'y_h']
    rel_errs_ml_queries = []
    print("Assessing on Queries...")
    for l1,s in zip(agg_label,labels):
        errs = 0
        print("Aggregate {} ===================================================".format(s))
        for no in no_queries:
            no = int(no)
            print("Queries : {}".format(no))
            test = test_df.sample(int(no*.2))
            train = train_df.sample(no)
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
            xgb = XGBRegressor()
            scaler = StandardScaler()
            scaler.fit(X_train)  # Don't cheat - fit only on training data
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)  # apply same transformation to test data

            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            if s=='SUM':
                print((y_pred==y_pred[0]).sum())
                print(((y_test==np.inf) | (y_test==-np.inf) | (y_test==np.nan)).sum())
            try:
                rel_error_ML_sum = np.median(np.abs(y_pred-y_test)/y_test)
                nrmsd = np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.std(y_test)
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                mae = metrics.median_absolute_error(y_test, y_pred)
                rel_errs_ml_queries.append([no, s, rmse, mae, nrmsd, rel_error_ML_sum])
            except ValueError:
                errs+=1
        print("Errors on {} of queries".format(errs/Config.chicago_queries))
    print("Saving Files...")
    eval_df = pd.DataFrame(rel_errs_ml_queries, columns=['queries', 'aggregate','rmse','mae', 'nrmsd', 'rel_error_median'])
    eval_df.to_csv('output/accuracy/csvs/chicago_assessment_on_queries_{}_queries.csv'.format(Config.chicago_queries))


if __name__=='__main__':
    np.random.seed(0)
    load_data()
    assess_on_queries()
