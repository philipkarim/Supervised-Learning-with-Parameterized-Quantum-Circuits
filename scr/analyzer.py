#In this script the plotting and analyzing part happens
import numpy as np

U_train=["trainU0_25", "trainU0_1", "trainU0_01", "trainU0_001"]
U_test=["testU0_25", "testU0_1", "testU0_01", "testU0_001"]

N_train=["trainN0_25", "trainN0_1", "trainN0_01", "trainN0_001"]
N_test=["testN0_25", "testN0_1", "testN0_01", "testN0_001"]

names=[nn, ss]

train_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/"+U_train[0]+".npy")
train_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/"+U_train[1]+".npy")
train_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/"+U_train[2]+".npy")
train_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/"+U_train[3]+".npy")

train_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/"+U_test[0]+".npy")
train_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/"+U_test[1]+".npy")
train_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/"+U_test[2]+".npy")
train_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/"+U_test[3]+".npy")
