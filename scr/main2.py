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
from sklearn.utils import shuffle

#Import the parameterized quantum circuit class
from PQC import QML
from optimize_loss import optimize
from utils import *

#Handling the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
idx = np.where(y < 2) # we only take the first two targets.

X = X[idx,:]
X=np.squeeze(X, axis=0)
y = y[idx]

X, y = shuffle(X, y, random_state=0)

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
scaler2 = MinMaxScaler()
scaler2.fit(X_train)
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

def train(circuit, n_epochs, n_batch_size, initial_thetas,lr, X_tr, y_tr, X_te, y_te):
    #Creating optimization object
    optimizer=optimize(lr, circuit)
    #Splits the dataset into batches
    batches=len(X_tr)//n_batch_size
    #Adds another batch if it has a reminder
    if len(X_tr)%n_batch_size!=0:
        batches+=1
    #Reshapes the data
    X_reshaped=np.reshape(X_tr,(batches,n_batch_size,X_tr.shape[1]))
    theta_params=initial_thetas.copy()

    #Defines a list containing all the prediction for each epoch
    prediction_epochs_train=[]
    loss_train=[]
    accuracy_train=[]

    prediction_epochs_test=[]
    loss_test=[]
    accuracy_test=[]

    temp_list=[]
    #Train parameters
    for epoch in range(n_epochs):
        #print(f"Epoch:{epoch}")
        for batch in range(batches):
            #print(f"Batch:{batch}")
            batch_pred=circuit.predict(X_reshaped[batch],theta_params)
            temp_list+=batch_pred
            theta_params=optimizer.gradient_descent(theta_params, batch_pred, y_tr[batch:batch+n_batch_size], X_reshaped[batch])
        
        #Computes loss and predicts on the test set with the new parameters
        train_loss=optimizer.binary_cross_entropy(temp_list,y_tr)
        test_pred=circuit.predict(X_te,theta_params)
        test_loss=optimizer.binary_cross_entropy(test_pred,y_te)
        
        #Tresholding the probabillities into hard label predictions
        temp_list=hard_labels(temp_list, 0.5)
        test_pred=hard_labels(test_pred,0.5)

        #Computes the accuracy scores
        acc_score=accuracy_score(y_tr,temp_list)
        acc_score_test=accuracy_score(y_te, test_pred)
        
        print(f"Epoch: {epoch}, loss:{train_loss}, accurcay:{acc_score}")
        #Saving the results
        loss_train.append(train_loss)
        accuracy_train.append(acc_score)
        prediction_epochs_train.append(temp_list)
        loss_test.append(test_loss)
        accuracy_test.append(acc_score_test)
        prediction_epochs_test.append(test_pred)

        temp_list.clear()

    return loss_train, accuracy_train, loss_test, accuracy_test

n_params=8
learning_rate=1.0
batch_size=1
init_params=np.random.uniform(0.25,0.75,size=n_params)
#init_params=np.arange(n_params)

epochs=50
qc=QML(0,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)

#Shuffle the data, print by accuracy score

train_loss, train_accuracy, test_loss, test_accuracy =train(qc, epochs, batch_size, 
                                                            init_params, learning_rate, X_tr=X_train,
                                                            y_tr=y_train, X_te=X_test,y_te=y_test)


#Analyze results



#Next steps:
"""
Things to try to fix the training:
-Switch up the ansatz, create an easier one and switch up the entanglement
-Have a look at the loss function, maybe implement the one in the project, scikit loss?
-Okay, should I try to sigmoid the function?

Do this now: Fix another ansats, put it in the descriptin, change loss function to binary cross,
             have a look at the derivative og the binary cross

-Try training the thing
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
-Some dead neurons, test with it
-For each computed training epoch, use the same parameters on testing? for each batch even?

Mean training loss of all batches for a specific epoch
Then use the same output parameters on the test set


Reoport:
error and accuracy vs batch size
error and accuracy vs epoch
lr vs accuracy or error



Appendix: derivative of gradient
"""

#Notes to self
#Compute loss, the deriavtive in gradient descent or adam is computed by evaluate the cirquit twice pi/2,
#Did I normalize the circuit between 0 and 2pi?