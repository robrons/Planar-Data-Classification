import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# Loaded data

X, Y = load_planar_dataset()

#defined layer sizes 
def layer_sizes(X, Y):

    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0] 

    return (n_x, n_h, n_y)


#initalised parameters 
def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2) 

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def forward_propagation(X, parameters):
   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache



def compute_cost(A2, Y, parameters):
   
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = np.sum(logprobs)
    
    cost = -np.squeeze(cost)/m     
    assert(isinstance(cost, float))
    
    return cost

 def backward_propagation(parameters, cache, X, Y):
   
    m = X.shape[1]
    
   
    W1 = parameters["W1"]
    W2 = parameters["W2"]
  
    A1 = cache["A1"]
    A2 = cache["A2"]
   
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads