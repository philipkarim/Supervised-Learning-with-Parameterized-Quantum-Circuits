#In this script the plotting and analyzing part happens
import numpy as np
from utils import plotter
#import matplotlib as plt
import matplotlib.pyplot as plt

U_train=["trainU0_25", "trainU0_1", "trainU0_01", "trainU0_001"]
U_test=["testU0_25", "testU0_1", "testU0_01", "testU0_001"]

N_train=["trainN0_25", "trainN0_1", "trainN0_01", "trainN0_001"]
N_test=["testN0_25", "testN0_1", "testN0_01", "testN0_001"]


train_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_train[0]+".npy")
train_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_train[1]+".npy")
train_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_train[2]+".npy")
train_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_train[3]+".npy")

test_a0_U0_25=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_test[0]+".npy")
test_a0_U0_1=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_test[1]+".npy")
test_a0_U0_01=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_test[2]+".npy")
test_a0_U0_001=np.load("Results/saved_data/iris/ansatz_0/distributions/"+U_test[3]+".npy")

train_a0_N0_25=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_train[0]+".npy")
train_a0_N0_1=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_train[1]+".npy")
train_a0_N0_01=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_train[2]+".npy")
train_a0_N0_001=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_train[3]+".npy")

test_a0_N0_25=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_test[0]+".npy")
test_a0_N0_1=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_test[1]+".npy")
test_a0_N0_01=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_test[2]+".npy")
test_a0_N0_001=np.load("Results/saved_data/iris/ansatz_0/distributions/"+N_test[3]+".npy")

def plott_distribution():
    plotter(train_a0_U0_25, "(0, 0.25)",  train_a0_U0_1, "(0, 0.1)", train_a0_U0_01, "(0, 0.01)", train_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_U0_25)), x_label="Epochs", y_label="Loss")
    plotter(test_a0_U0_25, "(0, 0.25)",  test_a0_U0_1, "(0, 0.1)", test_a0_U0_01, "(0, 0.01)", test_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_U0_25)), x_label="Epochs", y_label="Loss")

    plotter(train_a0_N0_25, "(0, 0.25)",  train_a0_N0_1, "(0, 0.1)", train_a0_N0_01, "(0, 0.01)", train_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_N0_25)), x_label="Epochs", y_label="Loss")
    plotter(test_a0_N0_25, "(0, 0.25)",  test_a0_N0_1, "(0, 0.1)", test_a0_N0_01, "(0, 0.01)", test_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_N0_25)), x_label="Epochs", y_label="Loss")
    return


def investigate_lr_nparam(ansatz, test_train):
    plt.rcParams.update({'font.size':12})
    #plt.rcParams['mathtext.fontset']='stix'
    #plt.rcParams['font.family']='STIXGeneral'
    #plt.rcParams['xtick.labelsize'] = 2
    #plt.rcParams['ytick.labelsize'] = 2
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams["figure.figsize"] = (12,5)


    rates = [0.1, 0.01]
    Ns = [10,14,18,22, 26]

    colors = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:olive"]
    ticks = ["*","s",".","D"]


    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    #fig1=plt.figure()
    #ax1=fig1.add_subplot(111)
    highest_index=0
    start=16

    for N in range(len(Ns)):
        for rate in range(len(rates)):
            filename_loss = "Results/saved_data/iris/ansatz_"+str(ansatz)+"/lr_params/"+test_train+"_lr%s_n%s.npy" %(rates[rate], Ns[N])
            #print(filename_loss)
            loss = np.fromfile(filename_loss)
            print(len(loss))
            print(loss)
            loss = loss[start:len(loss)+start]
            print(len(loss))
            print(loss)
            #Tresholding the loss
            #print(loss)
            #Tresholding the loss
            """
            for i in range(len(loss)):
                if loss[i]<0 or loss[i]>1:
                    loss[i]=0
                    if i>highest_index:
                        highest_index=i
            """
            iterations = np.arange(len(loss))
            ax2.plot(iterations,loss,color=colors[N],marker=ticks[rate],label="N=%s;Rate=%s" %(Ns[N],rates[rate]))
            #ax1.plot(iterations2,loss[highest_index+1:],color=colors[N],marker=ticks[rate],label="N=%s;Rate=%s" %(Ns[N],rates[rate]))

    ax2.legend(prop={'size':30})
    ax2.set_xlabel("Epochs",fontsize=12)
    ax2.set_ylabel("Loss",fontsize=12)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.92, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return


def optimal_run():
    


investigate_lr_nparam(0, "train")

