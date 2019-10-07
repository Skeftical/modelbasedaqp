import numpy as np
import xgboost as xgb
import os
import sys
os.chdir('../')
sys.path.append('.')
from sql_parser.parser import QueryVectorizer

class MLAF:
    """
    Wrapper for ML models that act as estimator of aggregate functions
    """

    def predict_one(self, attr_dict):
        array = []
        for k in self.features:
            array.append(np.float(attr_dict.get(k, np.nan)))
    #     temp.append(m.predict_one(attr))
        query_vector = xgb.DMatrix(np.array(array).reshape(1,-1), feature_names=self.features)
        return float(self.estimator.predict(query_vector))

    def predict_many(self, attr_dict):
        qv = QueryVectorizer(self.features, SET_OWN=True)
        for a in attr_dict:
            qv.insert(a, attr_dict[a])
        query_matrix = xgb.DMatrix(qv.to_matrix(), feature_names=self.features)
        return self.estimator.predict(query_vector)

    def __init__(self, estimator, rel_error, feature_names, af):
        self.estimator = estimator
        self.AF = af
        self.rel_error = rel_error
        self.features = feature_names


if __name__=="__main__":
    est = MLAF(None,0.33,['f1_lb','f1_ub','f2_lb','f3_lb'],'count')
    print(est.features)
    est.predict_many({'f1_lb': 1, 'f2_lb':2, 'f3_lb':[4,45,6]})
