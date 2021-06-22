#This import is just because of some duplicate of mpi or armadillo on the computer
import os
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
print(X_test)
print(X_test_scaled)

#Or use this method:
scaler = MinMaxScaler();  
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_test)
print(X_test_scaled)

#Training
n_parameters=4
qc=QML(0, X.shape[1],1,n_parameters, backend="qasm_simulator", shots=1024)

#Initialisation of thetas
initial_thetas = np.random.uniform(size=n_parameters)

predictions=qc.predict(X_train,initial_thetas)




#Next steps:
"""
-Try training the thing
    -Write loss function or find it in scikit
    -Write an optimizer, using adam or gd hopefully from Scikit
    -Need the derivative for this, use the parameter shift rule
-Try adding epochs and such to see how much the loss/accuracy is
-Maybe try plotting as a function of parameters, and play with initialization params of theta
-Maybe spread out the ansatz making it look better
-Add another ansatz, cool one in lin 6 bookmark
-try both ansatzes on breast cancer dataset
"""