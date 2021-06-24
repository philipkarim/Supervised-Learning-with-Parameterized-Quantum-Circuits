#This import is just because of some duplicate of mpi or armadillo on the computer
import os
from types import coroutine
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Importing packages 
import numpy as np
import matplotlib.pyplot as plt

#Importing qiskit
import qiskit as qk
from qiskit.visualization import *
from quantumnet import *

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

#Import the parameterized quantum circuit class
from PQC import QML


#Handling the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
idx = np.where(y < 2) # we only take the first two targets.

X = X[idx,:]
X=np.squeeze(X, axis=0)
y = y[idx]

#X, X_test, y, y_test = train_test_split(X,y,test_size=0.92,stratify=y)
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# Scaling the data using the scikit learn modules
scaler = StandardScaler();  
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print(X_test)
#print(X_test_scaled)

#Or use this method:
scaler2 = MinMaxScaler();  
scaler2.fit(X_train)
X_train_scaled = scaler2.transform(X_train)
X_test_scaled = scaler2.transform(X_test)
#print(X_test)
#print(X_test_scaled)

#Training
n_parameters=4
qc=QML(0, X.shape[1],1,n_parameters, backend="qasm_simulator", shots=1024)

#Initialisation of thetas
initial_thetas = np.random.uniform(size=n_parameters)
predictions=qc.predict(X_train,initial_thetas)

#for pr in range(len(y_test)):
#    print(predictions[pr],y_test[pr])


def cross_entropy(preds, targets, classes=2, epsilon=1e-12):
    """
    Computes cross entropy between the true labels and the predictions
    
    Args:
        preds:   predictions as an array or list
        targets: true labels as an array or list  
    
    Returns: loss as a scalar
    """
    #Creates matrixes to use one hot encoded labels
    distribution_preds=np.zeros((len(preds), classes))
    distribution_target=np.zeros((len(targets), classes))

    #Just rewriting the predictions and labels
    for i in range(len(preds)):
        distribution_preds[i][0]=1-preds[i]
        distribution_preds[i][1]=preds[i]
        
        if targets[i]==0:
            distribution_target[i][0]=1
        elif targets[i]==1:
            distribution_target[i][1]=1

    distribution_preds = np.clip(distribution_preds, epsilon, 1. - epsilon)
    n_samples = len(preds)
    ce = -np.sum(distribution_target*np.log(distribution_preds+1e-9))/n_samples
    return ce

"""
predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
ans = 0.71355817782  #Correct answer
x = cross_entropy(predictions, targets)
print(np.isclose(x,ans))

print(x)
"""
zz=cross_entropy(predictions, y_train)
print(zz)
#Next steps:
"""
-Try training the thing
    -Write an optimizer, using adam or gd hopefully from Scikit
    -Need the derivative for this, use the parameter shift rule
    -Explore with the batch size, batch size is how many samples 
    that will be predicted before the gradient descent does one loop. 1 loop only? i think so
-Try adding epochs and such to see how much the loss/accuracy is
-Maybe try plotting as a function of parameters, and play with initialization params of theta
-Maybe spread out the ansatz making it look better
-Add another ansatz, cool one in lin 6 bookmark
-try both ansatzes on breast cancer dataset
"""

#Notes to self
#Compute loss, the deriavtive in gradient descent or adam is computed by evaluate the cirquit twice pi/2,
#Did I normalize the circuit between 0 and 2pi?