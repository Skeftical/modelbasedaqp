import numpy as np

class MLAF:
    """
    Wrapper for ML models that act as estimator of aggregate functions
    """

    def predict_one(self, attr_dict):
        for k in attr_dict:
            self.vector_dict[k] = attr_dict[k]
        query_vector = np.array(self.vector_dict.values())
        return self.estimator.predict(query_vector)

    def __init__(self, estimator, aggregate_name, rel_error, feature_names):
        self.estimator = estimator
        self.aggregate_name = aggregate_name
        self.rel_error = rel_error
        self.vector_dict = { key : np.nan for key in feature_names}
        print(self.vector_dict.values())


if __name__=="__main__":
    est = MLAF(None, 'count', 0.33,['f1','f2','f3'])
