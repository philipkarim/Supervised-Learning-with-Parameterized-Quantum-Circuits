#In this script the plotting and analyzing part happens
import numpy as np
from utils import plotter

U_train=["trainU0_25", "trainU0_1", "trainU0_01", "trainU0_001"]
U_test=["testU0_25", "testU0_1", "testU0_01", "testU0_001"]

N_train=["trainN0_25", "trainN0_1", "trainN0_01", "trainN0_001"]
N_test=["testN0_25", "testN0_1", "testN0_01", "testN0_001"]


train_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/"+U_train[0]+".npy")
train_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/"+U_train[1]+".npy")
train_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/"+U_train[2]+".npy")
train_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/"+U_train[3]+".npy")

test_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/"+U_test[0]+".npy")
test_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/"+U_test[1]+".npy")
test_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/"+U_test[2]+".npy")
test_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/"+U_test[3]+".npy")

train_a0_N0_25=np.load("Results/saved_data/iris/ansatz_0/"+N_train[0]+".npy")
train_a0_N0_1=np.load("Results/saved_data/iris/ansatz_0/"+N_train[1]+".npy")
train_a0_N0_01=np.load("Results/saved_data/iris/ansatz_0/"+N_train[2]+".npy")
train_a0_N0_001=np.load("Results/saved_data/iris/ansatz_0/"+N_train[3]+".npy")

test_a0_N0_25=np.load("Results/saved_data/iris/ansatz_0/"+N_test[0]+".npy")
test_a0_N0_1=np.load("Results/saved_data/iris/ansatz_0/"+N_test[1]+".npy")
test_a0_N0_01=np.load("Results/saved_data/iris/ansatz_0/"+N_test[2]+".npy")
test_a0_N0_001=np.load("Results/saved_data/iris/ansatz_0/"+N_test[3]+".npy")


plotter(train_a0_U0_25, "(0, 0.25)",  train_a0_U0_1, "(0, 0.1)", train_a0_U0_01, "(0, 0.01)", train_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_U0_25)), x_label="Epochs", y_label="Loss")
plotter(test_a0_U0_25, "(0, 0.25)",  test_a0_U0_1, "(0, 0.1)", test_a0_U0_01, "(0, 0.01)", test_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_U0_25)), x_label="Epochs", y_label="Loss")

plotter(train_a0_N0_25, "(0, 0.25)",  train_a0_N0_1, "(0, 0.1)", train_a0_N0_01, "(0, 0.01)", train_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_N0_25)), x_label="Epochs", y_label="Loss")
plotter(test_a0_N0_25, "(0, 0.25)",  test_a0_N0_1, "(0, 0.1)", test_a0_N0_01, "(0, 0.01)", test_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_N0_25)), x_label="Epochs", y_label="Loss")