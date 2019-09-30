import numpy as np
import xgboost as xgb

class MLAF:
    """
    Wrapper for ML models that act as estimator of aggregate functions
    """

    def predict_one(self, attr_dict):
        array = []
        for k in self.features:
            array.append(np.float(attr[k]))
    #     temp.append(m.predict_one(attr))
        query_vector = xgb.DMatrix(np.array(array).reshape(1,-1), feature_names=self.features)
        return float(self.estimator.predict(query_vector))

    def __init__(self, estimator, rel_error, feature_names, af):
        self.estimator = estimator
        self.AF = af
        self.rel_error = rel_error
        self.features = feature_names


if __name__=="__main__":
    est = MLAF(None, 'count', 0.33,['f1','f2','f3'])
