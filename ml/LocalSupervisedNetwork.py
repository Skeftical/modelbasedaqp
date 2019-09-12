from copy import deepcopy,copy
import numpy as np
import warnings
import json

class PassToSystem(Exception):
    pass

class LocalSupervisedNetwork():
    """
    A network of local models. Utilizing both supervised and unsupervised ML Models.
    The specified 'partitioning algorithm' is used to partition the space into disjoint sets. The supervised
    'learning' algorithm is then used to train local models for each disjoint set
    """
    def fit(self, X, y):
        for x,y_ in zip(X,y):
            self.partial_fit(x,y_)

    def __trigger_training(self, ix):
        X = np.array(self.samples[ix])
        y = np.array(self.y_samples[ix])
        #Deepcopy the learning algorithm to initialize a new learner
        estimator = deepcopy(self.learning_algorithm)

        estimator.fit(X, y)

        #Associate node with estimator
        self.associations[ix] = estimator

    def __check_sample(self, ix):
        #In future work can be replaced with more advanced stopping criteria.
        #Like optimal stopping for decision making
        if len(self.samples[ix])==self.sample_threshold:
            self.__trigger_training(ix)

    def partial_fit(self, x, y):
        #Train unsupervised network of nodes
        self.partitioning_algorithm.partial_fit(x)
        #Based on the best node increment count for Node and add sample to samples
        ix = self.partitioning_algorithm.get_best_node(x)
        if ix not in self.counts.keys():
            self.counts[ix] = 1
            self.samples[ix] = [x]
            self.y_samples[ix] = [y]
        else:
            self.counts[ix]+=1
            self.samples[ix].append(x)
            self.y_samples[ix].append(y)
        #Check if samples have reached the threshold and train Supervised model
        self.__check_sample(ix)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.__predict_one(x))
        return np.array(predictions)

    def __get_from_many(self, x):
        print(self.associations)
        print(list(map(lambda y1 : float(y1.predict(np.array([x]))), list(self.associations.values()))))
        return np.mean(list(map(lambda y1 : float(y1.predict(np.array([x]))), list(self.associations.values()))))


    def __predict_one(self, x):
        ix = self.partitioning_algorithm.get_best_node(x)
        if ix not in self.associations.keys():
            if self.strategy=='Many':
                return self.__get_from_many(x.reshape(1,-1).squeeze())
            elif self.strategy=='None':
                # print("Closest {0} with no trained model passing query to system".format(ix))
                return None
        else:
            return float(self.associations[ix].predict(x.reshape(1,-1)))

    def __init__(self, partitioning_algorithm, learning_algorithm, sample_threshold=100, strategy='None'):
        """
        Parameters :

        partitioning_algorithm : Unsupervised clustering/partitioning/quantization algorithm. Needs to implement
        partial_fit function and predict function.

        learning algorithm : Supervised regression algorithm. Will be constructed for every disjoint set
        sample_threshold : Is the number of samples a set needs to have before training a learning algorithm over it.
        strategy : Strategy for dealing with missing learners in associations. ['None' | 'Many' | 'Closest Association']
        """
        self.strategy = strategy
        self.sample_threshold = sample_threshold
        self.partitioning_algorithm = partitioning_algorithm
        self.learning_algorithm = learning_algorithm
        #Dictionary to hold cluster nodes mapping to Supervised Models
        self.associations = {}
        #Need to hold the count of samples for each node
        self.counts = {}
        #Need to store samples for each node
        self.samples = {}
        self.y_samples = {}

class LocalSupervisedOfflineNetwork(object):
    def fit(self, X, y):
        #Cluster the dataset
        self.partitioning_algorithm.fit(X)
        #Find where each point belongs to
        memberships = self.partitioning_algorithm.predict(X)

        #For each cluster
        for k in np.unique(memberships):
            #identify the points associated with k
            self.samples[k] = copy(X[memberships==k])
            #and their target variables
            self.y_samples[k] = copy(y[memberships==k])
            #Update the counts of each cluster
            self.counts[k] = self.samples[k].shape[0]
            #Create the association of the cluster with the learning algorithm
            estimator = deepcopy(self.learning_algorithm)
            self.associations[k] = estimator.fit(self.samples[k], self.y_samples[k])



    def predict(self, X):
        assignments = self.partitioning_algorithm.predict(X)
        predictions = []
        c = {}
        if self.warnings_b:
            warnings.warn(str("Assignments {}".format(list(np.unique(assignments)))))
        for x,a in zip(X, assignments):
            if a not in c:
                c[a] = 1
            else:
                c[a]+=1
            predictions.append(self.associations[a].predict(x.reshape(1,-1)))
        # predictions = map(lambda a: self.associations[a].predict(X[assignments==a]) ,np.unique(assignments))
        # predictions = reduce(lambda x1,x2: np.concatenate((x1,x2)), predictions)
        if self.warnings_b:
            warnings.warn("Predictions were")
            warnings.warn(str(c))
        return np.array(predictions)




    def __init__(self, partitioning_algorithm, learning_algorithm, warnings_b=False):
            """
            Parameters :

            partitioning_algorithm : Unsupervised clustering/partitioning/quantization algorithm.
            learning algorithm : Supervised regression algorithm. Will be constructed for every disjoint set

            """
            self.warnings_b = warnings_b
            self.partitioning_algorithm = partitioning_algorithm
            self.learning_algorithm = learning_algorithm
            #Dictionary to hold cluster nodes mapping to Supervised Models
            self.associations = {}
            #Need to hold the count of samples for each node
            self.counts = {}
            #Need to store samples for each node
            self.samples = {}
            self.y_samples = {}
