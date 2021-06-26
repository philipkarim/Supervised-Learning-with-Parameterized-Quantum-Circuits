#This import is just because of some duplicate of mpi or armadillo on the computer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Importing packages 

from types import coroutine

import numpy as np
import matplotlib.pyplot as plt
import random

#Importing qiskit
import qiskit as qk
from qiskit.visualization import *

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.datasets import load_breast_cancer

#Import the parameterized quantum circuit class
from PQC import QML
from optimize_loss import optimize
from utils import *

#Seeding the program
random.seed(2021)

#Handling the data
#Choose a datset
dataset="iris"
#dataset="breastcancer"

if dataset=="iris":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    idx = np.where(y < 2) # we only take the first two targets.

    X = X[idx,:]
    X=np.squeeze(X, axis=0)
    y = y[idx]
    import collections
    dirname=collections
    #dirname=os.path.dirname()
    #print(dirname)
    path="Results/saved_data/iris/"
    """
    xx=np.array([1,1,2])
    np.save("sample.npy", xx)
    print(np.load("sample.npy"))
    np.save("Results/saved_arrays/test.npy", xx)
    print(path)
    """

elif dataset=="breastcancer":
    data = load_breast_cancer()
    X = data.data #features
    y = data.target #targets

    #Uses the first four feutures
    X=np.delete(X, np.s_[4:len(X[0])], axis=1) 
    path="Results/saved_data/breastcancer/"

else:
    print("No datset chosen\nClosing the program")
    quit()

X, y = shuffle(X, y, random_state=0)

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, stratify=y)

#Or use this method:
scaler2 = MinMaxScaler()
scaler2.fit(X_train)
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

#Set parameters
#Use: 10,0.1,1,0.01
n_params=10
learning_rate=0.01
batch_size=1
init_params=np.random.normal(0.,0.01,size=n_params)
ansatz=1
epochs=50
qc=QML(ansatz,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)
qc_2=QML(1,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)

#Choose type of investigation, if all parameters are chosen prehand, set both to false
regular_run=False
inspect_distribution=True
inspect_lr_param=False

def train(circuit, n_epochs, n_batch_size, initial_thetas,lr, X_tr, y_tr, X_te, y_te):
    #Creating optimization object
    optimizer=optimize(lr, circuit)
    #Splits the dataset into batches
    batches=len(X_tr)//n_batch_size
    #Adds another batch if it has a reminder
    if len(X_tr)%n_batch_size!=0:
        batches+=1
    #Reshapes the data into batches
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
            #print(batch_pred, y_tr[batch:batch+n_batch_size])
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
        
        print(f"Epoch: {epoch}, loss:{train_loss}, accuracy:{acc_score}")
        #Saving the results
        loss_train.append(train_loss)
        accuracy_train.append(acc_score)
        prediction_epochs_train.append(temp_list)
        loss_test.append(test_loss)
        accuracy_test.append(acc_score_test)
        prediction_epochs_test.append(test_pred)

        temp_list.clear()

    return loss_train, accuracy_train, loss_test, accuracy_test


def investigate_distribution(type, folder_name):
    """
    Function that investigate the best initialisation of the varational parameters
    and also tests if uniform- or normal distribution is best
    
    Needs to be ran at an unix system bases computer due to the fork commands

    args: 
            type: "U" or "N" for uniform or normal
    """
    pid = os.fork()
    if pid:
        train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.01, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
        np.save(data_path(folder_name, "train"+type+"0_01.npy"), np.array(train_loss))
        np.save(data_path(folder_name, "test"+type+"0_01.npy"), np.array(test_loss))

    else:
        pid1 = os.fork()
        if pid1:
            train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.1, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
            np.save(data_path(folder_name, "train"+type+"0_1.npy"), np.array(train_loss))
            np.save(data_path(folder_name, "test"+type+"0_1.npy"), np.array(test_loss))
        else:
            pid2=os.fork()
            if pid2:
                train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.25, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
                np.save(data_path(folder_name, "train"+type+"0_25.npy"), np.array(train_loss))
                np.save(data_path(folder_name, "test"+type+"0_25.npy"), np.array(test_loss))            
            else:
                train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.001, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
                np.save(data_path(folder_name, "train"+type+"0_001.npy"), np.array(train_loss))
                np.save(data_path(folder_name, "test"+type+"0_001.npy"), np.array(test_loss))
    
    return

def investigate_lr_params(folder_name, folder_name2, lr_list, n_params_list):
    """
    Function that investigate the best learning rate and learning rate
    
    Needs to be ran at an unix system bases computer due to the fork commands

    args: 
            folder_name:    place of saving the folder
            lr_list:        list of learning rates
            n_params_list:  list of number of parameters to run 
    """
    pid = os.fork()
    if pid:
        for i in lr_list:
            for j in n_params_list:
                train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, np.random.normal(0.,0.01,size=j), i, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
                np.save(data_path(folder_name, "lr_params/train_lr"+str(i)+"_n"+str(j)+".npy"), np.array(train_loss))
                np.save(data_path(folder_name, "lr_params/test_lr"+str(i)+"_n"+str(j)+".npy"), np.array(test_loss))

    else:
        for ii in lr_list:
            for jj in n_params_list:
                train_loss, train_accuracy, test_loss, test_accuracy=train(qc_2, epochs, batch_size, np.random.normal(0.,0.01,size=jj), ii, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
                np.save(data_path(folder_name2, "lr_params/train_lr"+str(ii)+"_n"+str(jj)+".npy"), np.array(train_loss))
                np.save(data_path(folder_name2, "lr_params/test_lr"+str(ii)+"_n"+str(jj)+".npy"), np.array(test_loss))
    
    return


if ansatz==0:
    folder="ansatz_0/"
else:
    folder="ansatz_1/"


file_folder=data_path(path,folder)

if inspect_distribution==True:
    file_folder=data_path(path,folder)
    investigate_distribution("N", file_folder)

elif inspect_lr_param==True:
    file_folder2=data_path(path,"ansatz_1/")
    #Learning rates and number of paraeters to investigate
    lrs=[0.1, 0.01, 0.001, 0.0001]
    n_par=[4, 8, 12, 16, 20, 24]
    lrs=[0.1]
    n_par=[4]
    investigate_lr_params(file_folder, file_folder2, lrs, n_par)

elif regular_run==True:
    train_loss, train_accuracy, test_loss, test_accuracy =train(qc, epochs, batch_size, 
                                                            init_params, learning_rate, X_tr=X_train,
                                                            y_tr=y_train, X_te=X_test,y_te=y_test)



#np.save('trainlosses.npy', np.array(trainlosses))




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

#-Make it ready to plot
-Fix the ansatz and batch maybe
-Plot learning rate and diff parameters with fork
-Choose the best lr and param
-Maybe explore the initialization also? gaussian/uniform, initialiation rate

-Plot train accuracy and train loss with test, beside or in same plot?



Reoport:
error and accuracy vs batch size
error and accuracy vs epoch
lr vs accuracy or error
accuracy as a function of parameters with different learning rates


Appendix: derivative of gradient
"""
