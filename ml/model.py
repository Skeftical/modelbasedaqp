import numpy as np
import xgboost as xgb

class MLAF:
    """
    Wrapper for ML models that act as estimator of aggregate functions
    """

    def predict_one(self, attr_dict):
        for k in attr_dict:
            self.vector_dict[k] = attr_dict[k]
        query_vector = np.array(list(self.vector_dict.values())).reshape(1,-1)
        if af!='count':
            query_vector = xgb.DMatrix(query_vector)
        return float(self.estimator.predict(query_vector))

    def __init__(self, estimator, rel_error, feature_names, af):
        self.estimator = estimator
        self.AF =
        self.rel_error = rel_error
        self.vector_dict = { key : np.nan for key in feature_names}
        print(self.vector_dict)


if __name__=="__main__":
    est = MLAF(None, 'count', 0.33,['f1','f2','f3'])
