import numpy as np
import warnings

class GrowingNetwork():
    """
    Growing Network implementation from the Algorithm in
    'A self-organising network that grows when required'
    """

    def get_best_node(self, x):
        return np.argmin(np.linalg.norm(self.A-x,axis=1))

    def _distance(self,x,w):
        '''Euclidean distance between two vectors'''
        return np.linalg.norm(x-w)

    def _create_connection(self, b1, b2):
        '''
        arg b1 : Closest node index
        arg b2 : Second-closest node index
        '''
        #If there is a connection
        if self.connections[b1,b2] and self.connections[b2,b1]:
            #Reset ages
            self.ages[b1,b2] = 0
            self.ages[b2,b1] = 0
        else: #Create Connection
            self.connections[b1,b2] = 1
            self.connections[b2,b1] = 1

    def _below_activity(self,x, b1):
        """
        Check if activity of best matching unit below threshold
        argument :
        x : Input Vector
        b1 : Index of best node
        """
        #Activity is 1 if distance is 0 and exponentially decreases as the distance increases
        w_b1 = self.A[b1]
        activity =  np.exp(-np.linalg.norm(x-w_b1))
        return activity < self.a

    def _below_firing(self,b1):
        return self.firing_vector[b1] < self.h

    def _add_new_node(self, b1, b2, x):
        """
        Creates a new node between b1 and b2. Where the weights for the new node are a function of the input vector
        and best matching unit. Inserts new edges and removes previous link
        argument:
        b1 : Index of best matching unit
        b2 : Index of second best matching unit
        x : Input Vector
        """
        w_b1 = self.A[b1]
        #Add new node to A
        weight_vector =  (w_b1+x)/2
        self.A = np.vstack((self.A, weight_vector))
        rows = self.connections.shape[0]
        #Append new row and column in connections to represent new node
        self.connections = np.column_stack((self.connections, np.zeros(rows)))
        columns = self.connections.shape[1]
        self.connections = np.row_stack((self.connections, np.zeros(columns)))
        #Same for ages
        self.ages = np.column_stack((self.ages, np.zeros(rows)))
        self.ages = np.row_stack((self.ages, np.zeros(columns)))
        #Add firing counter
        self.firing_vector = np.append(self.firing_vector, 1)

        #Add Connections between new node and b1 b2
        ix = self.connections.shape[0]-1 #Get index of latest node 'r'
        self._create_connection(b1, ix)
        self._create_connection(b2, ix)
        #Remove link between b1 b2
        self.connections[b1,b2] = 0
        self.connections[b2,b1] = 0


    def _best(self,x):
        """
        Return the indices of the two best nodes
        argument:
        x : Input node
        """
        b1,b2 = np.argsort(np.linalg.norm(self.A-x,axis=1))[:2]
        #Returned indices of two closest nodes
        self._create_connection(b1,b2) #Step 4
        return (b1,b2)

    def _get_neighbours(self, w):
        """
        argument:
        w: Node
        return:
        boolean vector with indexes of neighbours
        """
        b_neighbours = self.connections[w,:].astype(bool)
#         b_neighbours = map(lambda x : True if x else False, neighbours)
        return b_neighbours

    def _adapt(self, w, x):
        """
        argument :
        w : Index of winner node
        x : input vector
        """
        weight_vector = self.A[w]
        #Adapt winner node
        hs = self.firing_vector[w]
        #Calculate Delta
        delta = self.es*hs*(x - weight_vector)
        #Update
        self.A[w] += delta
        #Adapt neighbours
        b_neighbours = self._get_neighbours(w)
        w_neighbours = self.A[b_neighbours] # Matrix of nXl (n: neighoubrs, l: length of weight vector)
        hi = self.firing_vector[b_neighbours] # Vector of firing rates for neighbours
        delta = self.en * np.multiply(hi.reshape(-1,1),(x-w_neighbours))
        #Update
        self.A[b_neighbours]+=delta

    def _age(self, w):
        """
        argument:
        w: Winner node index
        """
        b_neighbours = self._get_neighbours(w)
        self.ages[w, b_neighbours]+=1
        self.ages[b_neighbours, w]+=1

    def _reduce_firing(self, w):
        """
        argument:
        w: Winner node index
        t: timestep
        """
        h0 = self.h0 #Initial strength
        S = self.S #Stimulus strengh usually 1
        #Rest are constants controlling the behavior of the curve
        ab = self.ab
        tb = self.tb
        an = self.an
        tn = self.tn
        #=================
        t = self.t #Timestep
        self.firing_vector[w] = h0 - S/ab*(1 - np.exp(-ab*t/tb))
        if self.warnings_b:
            warnings.warn('Firing Activity for {0} is {1}'.format(w, self.firing_vector[w]))
        b_neighbours = self._get_neighbours(w)
        self.firing_vector[b_neighbours] = h0 - S/an*(1- np.exp(-an*t/tn))


    def __init__(self, X, a=0.1, h=0.1, en=0.1, es=0.1, an=1.05, ab=1.05, h0=1, tb=3.33, tn=14.3, S=1, warnings_b=False):
        '''
        arg X: data points [n,m] ndarray
        arg a:  activity threshold
        arg h: firing threshold
        '''
        self.a = a
        self.h = h
        self.es = es
        self.en = en
        self.an = an
        self.ab = ab
        self.h0 = h0
        self.tb = tb
        self.tn = tn
        self.S = S
        self.t = 1 # Timestep
        self.warnings_b = warnings_b
        #Create weight vectors for initial nodes
        w1 = X[np.random.randint(X.shape[0])]
        w2 = X[np.random.randint(X.shape[0])]
        #Node matrix A
        self.A = np.array([w1, w2])
        #Only 2 nodes available at the beginning
        self.connections = np.zeros((2,2)) # Matrix nxn (n=|nodes|) of 0,1 to indicated connection
        self.ages = np.zeros((2,2))
        self.firing_vector = np.ones(2)

    def fit(self, X):
        """
        Fit a dataset using GWR
        Caution : Reinitializes if a previous fit was made
        arg X : X nd-array, [samples, features]
        """
        self.__init__(X, self.a, self.h)
        for x in X: #In algorithm this is step 1
            b1,b2 = self._best(x) #step 2
            if self._below_activity(x, b1) and self._below_firing(b1): #Step 5,6
        #         print("Below activity and firing threshold")
                #New node should be added
                self._add_new_node(b1,b2,x) # Step 6
            else :
        #         print("Activity : {0}\tFiring : {1}".format(gw._below_activity(x,b1), gw._below_firing(b1)))
                #Adapting current nodes
                self._adapt(b1, x) #Step 7
                self._age(b1) # Step 8
                self._reduce_firing(b1) #Step 9
            #Increase timestep
            self.t+=1
        return self

    def partial_fit(self, x):
        """
        Fit a single example
        arg x : single vector [1, features]
        """
        b1,b2 = self._best(x) #step 2
        if self._below_activity(x, b1) and self._below_firing(b1): #Step 5,6
    #         print("Below activity and firing threshold")
            #New node should be added
            self._add_new_node(b1,b2,x) # Step 6
        else :
    #         print("Activity : {0}\tFiring : {1}".format(gw._below_activity(x,b1), gw._below_firing(b1)))
            #Adapting current nodes
            self._adapt(b1, x) #Step 7
            self._age(b1) # Step 8
            self._reduce_firing(b1) #Step 9
        #Increase timestep
        self.t+=1
        return self

    def predict(self, X):
        """
        Predict the nodes where the samples are closest too. (Most similar to)
        arg X : ndarray, shape [samples, features]
        returns : n-array, of [samples] with the index of each node for each sample
        """
        predictions = []
        for x in X:
            prediction = np.argsort(np.linalg.norm(np.subtract(self.A,x),axis=1))[0]
            predictions.append(prediction)
        return np.array(predictions)
