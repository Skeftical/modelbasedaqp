import numpy as np
import warnings
from sklearn.cluster import MiniBatchKMeans

class GrowingMiniBatchKMeans(MiniBatchKMeans):

    def get_best_node(self, x):
        if hasattr(self, 'cluster_centers_'):
            return np.argmin(np.linalg.norm(self.cluster_centers_-x,axis=1, \
            ord=self.ord))
        else:
            return 0

    def __get_closest_distance(self, x):
        return np.min(np.linalg.norm(self.cluster_centers_-x,axis=1, \
        ord=self.ord))

    def __create_new_node(self, x):
        warnings.warn('Initializing New Node ')
        self.cluster_centers_ = np.vstack((self.cluster_centers_, x))
        self.n_clusters+=1
        self.counts_ = np.append(self.counts_,0)

    def partial_fit(self, x):
        """
        Fits one sample at a time. If closest node distance is larger than treshold then create new node
        """
        if hasattr(self, 'cluster_centers_'):
            distance = self.__get_closest_distance(x)
            if distance>self.threshold:
                self.__create_new_node(x)
            else:
                super(GrowingMiniBatchKMeans, self).partial_fit(x.reshape(1,-1))
        else:
            super(GrowingMiniBatchKMeans, self).partial_fit(x.reshape(1,-1))


    def __init__(self, threshold=0.3, order=2, verbose=False):
        """
        args:
        threshold = Threshold for minimum distance between points
        ord = norm order, default euclidean norm (=2)
        """
        self.verbose = verbose
        self.threshold = threshold
        self.ord = order
        super(GrowingMiniBatchKMeans, self).__init__(n_clusters=1, verbose=self.verbose)
