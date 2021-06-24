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

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

#Import the parameterized quantum circuit class
from PQC import QML
from optimize_loss import optimize

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

"""
# Scaling the data using the scikit learn modules
scaler = StandardScaler();  
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print(X_test)
#print(X_test_scaled)
"""

#Or use this method:
scaler2 = MinMaxScaler();  
scaler2.fit(X_train)
X_train_scaled = scaler2.transform(X_train)
X_test_scaled = scaler2.transform(X_test)
#print(X_test_scaled[0])
#print(X_test)
#print(X_test_scaled)


#for pr in range(len(y_test)):
#    print(predictions[pr],y_test[pr])



def train(circuit, n_epochs, n_batch_size, initial_thetas,lr, X_train=X_train, y_train=y_train):
    #Creating optimization object
    optimizer=optimize(lr, circuit)
    #Splits the dataset into batches
    batches=len(X_train)//n_batch_size
    #Adds another batch if it has a reminder
    if len(X_train)%n_batch_size!=0:
        batches+=1
    #Reshapes the data
    #print(len(X_train))
    X_reshaped=np.reshape(X_train,(batches,n_batch_size,X_train.shape[1]))
    #print(X_reshaped)
    #print(X_reshaped[0])

    theta_params=initial_thetas

    #Defines a list containing all the prediction for each epoch
    prediction_epochs_train=[]
    loss_train=[]
    accuracy_train=[]

    temp_list=[]
    #Train parameters
    for epoch in range(n_epochs):
        print(f"Epoch:{epoch}")
        for batch in range(batches):
            print(f"Batch:{batch}")
            batch_pred=circuit.predict(X_reshaped[batch],theta_params)
            temp_list+=batch_pred
            theta_params=optimizer.gradient_descent(theta_params, batch_pred, y_train[batch:batch+n_batch_size], X_reshaped[batch])
            #print(theta_params)

        prediction_epochs_train.append(temp_list)
        temp_list.clear()

    
            #Add result to a main list of outputs
            #Compute loss with the whole list after each epoch and add to list

    return theta_params, prediction_epochs_train, loss_train, accuracy_train

n_params=4
learning_rate=0.1
batch_size=5
init_params=np.random.uniform(0,0.01,size=n_params)
epochs=5
qc=QML(0,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)


test_parameters, train_predictions, train_loss, train_accuracy=train(qc, epochs, batch_size, init_params, learning_rate, X_train=X_train_scaled, y_train=y_train)


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
-Try to use an entanglement encoder which is a cnot but on all cirquits in separate thing
-Explore best epoch and batch size
-Choose batch and epoch from test or train or validation set?
-Normalized, so the largest output is 1 and smallest output is 0,
therefor softmax or sigmoid is not nessecary.
-Remember to seed the initialization theta

Reoport:
error and accuracy vs batch size
error and accuracy vs epoch
lr vs accuracy or error



Appendix: derivative of gradient
"""

#Notes to self
#Compute loss, the deriavtive in gradient descent or adam is computed by evaluate the cirquit twice pi/2,
#Did I normalize the circuit between 0 and 2pi?