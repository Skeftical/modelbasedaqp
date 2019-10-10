import pandas as pd
import numpy as np
#ML Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Models
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
#Metrics
from sklearn import metrics
import sys
import os
os.chdir("../../")
sys.path.append('.')
from confs import Config


if not os.path.exists('output/accuracy'):
        print('creating ', 'accuracy dir')
        os.makedirs('output/accuracy')
if not os.path.exists('output/accuracy/csvs'):
        print('creating ', 'accuracy csvs')
        os.makedirs('output/accuracy/csvs')
# mars = Earth()
ridge = Ridge()
sgd = SGDRegressor(tol=1e-3)
svr = SVR(kernel='rbf', gamma='auto')
xgb = XGBRegressor()
model = [ridge, sgd, svr, xgb]
model_str = ['Ridge', 'SGD', 'SVR', 'XGB']

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def assess_on_models():
    errors = []
    predicates = Config.predicates
    cols = Config.columns
    aggregate_str = Config.aggregates
    rmse_results = []
    for pred in predicates:
        for col in cols:
            if col<=pred:
                continue;
            print("Predicates {0} || Columns {1}".format(pred, col))
            workload = np.loadtxt('input/synthetic_workloads/{}-Queries/query-workload-predicates_{}-cols_{}.csv'.format(Config.queries,pred, col), delimiter=',')
            workload = workload[~np.isnan(workload).any(axis=1)]
            if workload.shape[0]<0.1*Config.queries:
                print("Error on workload possibly containing large fraction of nans : {}".format(1-workload.shape[0]/Config.queries))
                errors.append('query-workload-predicates_{}-cols_{}.csv'.format(pred, col))
                continue;
            aggregate = range(workload.shape[1]-5,workload.shape[1])
            for t_y, l_Y in zip(aggregate, aggregate_str):
                X_train, X_test, y_train, y_test = train_test_split(
                     workload[:,:workload.shape[1]-5], workload[:,t_y], test_size=0.3, random_state=0)
                # X_train[(X_train==1e-8) | (X_train==1e+8)] = np.mean(X_train)
                # X_test[(X_test==1e-8) | (X_test==1e+8)] = np.mean(X_test)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)  # apply same transformation to test data
                for m, m_l in zip(model, model_str):
    #                 print("\tFitting for Agg {0} with {1}".format(l_Y, m_l))
                    m.fit(X_train, y_train)
                    predictions_test = m.predict(X_test)
                    ml_relative_error = np.mean(np.abs((y_test - predictions_test)/y_test))
                    ml_relative_error_median = np.median(np.abs((y_test - predictions_test)/y_test))
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))
                    mae = metrics.median_absolute_error(y_test, predictions_test)
                    nrmsd = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))/np.std(y_test)
                    rmse_results.append([pred, col, m_l, l_Y, rmse, nrmsd, mae, ml_relative_error, ml_relative_error_median])
    if len(errors)!=0:
        print("Finished with errors on:")
        for e in errors:
            print(e)
    test_df = pd.DataFrame(rmse_results, columns=['predicates', 'columns', 'model', 'aggregate', 'rmse','nrmsd','mae','rel_error_mean', 'rel_error_median'])


    test_df.to_csv('output/accuracy/csvs/synthetic_workloads_eval_on_models_{}_queries.csv'.format(Config.queries))

def __compute_ratio_of_non_outliers(d):
        Q3 = np.percentile(d,.75)
        Q1 = np.percentile(d, .25)
        IQR = Q3-Q1
        cutoff_points= (Q1-1.5*IQR, Q3+1.5*IQR)
        non_outliers = sum(((d>= cutoff_points[0]) & (d<= cutoff_points[1])).astype(int))
        return (non_outliers/float(d.shape[0]))

def assess_on_workloads():
    errors = []
    predicates = Config.predicates
    cols = Config.columns
    aggregate_str = Config.aggregates
    relative_errors = []
    for pred in predicates:
        for col in cols:
            if col<=pred:
                continue;
            print("Predicates {0} || Columns {1}".format(pred, col))

            workload = np.loadtxt('input/synthetic_workloads/{}-Queries/query-workload-predicates_{}-cols_{}.csv'.format(Config.queries,pred, col), delimiter=',')
            workload = workload[~np.isnan(workload).any(axis=1)]
            if workload.shape[0]<0.5*Config.queries:
                print("Error on workload possibly containing large fraction of nans : {}".format(1-workload.shape[0]/Config.queries))
                errors.append('query-workload-predicates_{}-cols_{}.csv'.format(pred, col))
                continue;
            aggregate = range(workload.shape[1]-5,workload.shape[1])
            for t_y, l_Y in zip(aggregate, aggregate_str):
                non_outliers_ratio = __compute_ratio_of_non_outliers(workload[:,t_y])
                xgb = XGBRegressor()
                X_train, X_test, y_train, y_test = train_test_split(
                     workload[:,:workload.shape[1]-5], workload[:,t_y], test_size=0.2, random_state=0)

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)  # apply same transformation to test data

                xgb.fit(X_train, y_train)
                predictions_training = xgb.predict(X_train)

                predictions_test = xgb.predict(X_test)
                assert y_test.shape == predictions_test.shape
                ml_relative_error = np.mean(np.abs((y_test - predictions_test)/y_test))
                ml_relative_error_median = np.median(np.abs((y_test - predictions_test)/y_test))
                rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))
                mae = metrics.median_absolute_error(y_test, predictions_test)
                nrmsd = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))/np.std(y_test)
                relative_errors.append([pred, col, rmse, mae, ml_relative_error, ml_relative_error_median, nrmsd,  l_Y, non_outliers_ratio])

    if len(errors)!=0:
        print("Finished with errors on:")
        for e in errors:
            print(e)
    eval_df = pd.DataFrame(relative_errors, columns=['predicates', 'columns','rmse','mae', 'relative_error_mean', 'relative_error_median', 'nrmsd', 'aggregate' , 'non_outliers_ratio'])
    eval_df.to_csv('output/accuracy/csvs/synthetic_workloads_eval_on_workloads_{}_queries.csv'.format(Config.queries))


def assess_on_no_queries():
    errors = []
    predicates = Config.predicates
    cols = Config.columns
    aggregate_str = Config.aggregates
    queries_number = np.linspace(0.1,1,10)*(Config.queries/10)
    relative_errors = []
    for n in queries_number:
        n = int(n)
        print("Number of Queries {0}".format(n))
        for pred in predicates:
            for col in cols:
                if col<=pred:
                    continue;
                print("Predicates {0} || Columns {1}".format(pred, col))

                workload = np.loadtxt('input/synthetic_workloads/{}-Queries/query-workload-predicates_{}-cols_{}.csv'.format(Config.queries,pred, col), delimiter=',')
                workload = workload[~np.isnan(workload).any(axis=1)]
                if workload.shape[0]<0.1*Config.queries:
                    print("Error on workload possibly containing large fraction of nans : {}".format(1-workload.shape[0]/Config.queries))
                    errors.append('query-workload-predicates_{}-cols_{}.csv'.format(pred, col))
                    continue;
                workload = workload[:n,:]
                aggregate = range(workload.shape[1]-5,workload.shape[1])
                for t_y, l_Y in zip(aggregate, aggregate_str):
                    non_outliers_ratio = __compute_ratio_of_non_outliers(workload[:,t_y])
                    xgb = XGBRegressor()
                    X_train, X_test, y_train, y_test = train_test_split(
                         workload[:,:workload.shape[1]-5], workload[:,t_y], test_size=0.2, random_state=0)

                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)  # apply same transformation to test data

                    xgb.fit(X_train, y_train)
                    predictions_training = xgb.predict(X_train)
            #         print("Training RMSE {0}".format(np.sqrt(metrics.mean_squared_error(y_train, predictions_training))))
                    predictions_test = xgb.predict(X_test)
                    ml_relative_error = np.mean(np.abs((y_test - predictions_test)/y_test))
                    ml_relative_error_median = np.median(np.abs((y_test - predictions_test)/y_test))
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))
                    mae = metrics.median_absolute_error(y_test, predictions_test)
                    nrmsd = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))/np.std(y_test)
                    relative_errors.append([pred, col,n, rmse, mae, ml_relative_error, ml_relative_error_median,nrmsd, l_Y, non_outliers_ratio])
    if len(errors)!=0:
            print("Finished with errors on:")
            for e in errors:
                print(e)
    eval_df = pd.DataFrame(relative_errors, columns=['predicates', 'columns','queries', 'rmse','mae', 'relative_error_mean', 'relative_error_median','nrmsd', 'aggregate' , 'non_outliers_ratio'])
    eval_df.to_csv('output/accuracy/csvs/synthetic_workloads_eval_on_workloads_varying_queries_{}_queries.csv'.format(Config.queries))

if __name__=='__main__':
    np.random.seed(0)
    assess_on_models()
    assess_on_workloads()
    assess_on_no_queries()
