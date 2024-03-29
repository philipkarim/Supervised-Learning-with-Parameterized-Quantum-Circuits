#In this script the plotting and analyzing part happens
import numpy as np
from utils import plotter
#import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

def plott_distribution():
    #This part can be done in a much mores sophisticated way, rewrite if time
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

    plotter(train_a0_U0_25, "(0, 0.25)",  train_a0_U0_1, "(0, 0.1)", train_a0_U0_01, "(0, 0.01)", train_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_U0_25)), x_label="Epochs", y_label="Loss")
    plotter(test_a0_U0_25, "(0, 0.25)",  test_a0_U0_1, "(0, 0.1)", test_a0_U0_01, "(0, 0.01)", test_a0_U0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_U0_25)), x_label="Epochs", y_label="Loss")

    plotter(train_a0_N0_25, "(0, 0.25)",  train_a0_N0_1, "(0, 0.1)", train_a0_N0_01, "(0, 0.01)", train_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(train_a0_N0_25)), x_label="Epochs", y_label="Loss")
    plotter(test_a0_N0_25, "(0, 0.25)",  test_a0_N0_1, "(0, 0.1)", test_a0_N0_01, "(0, 0.01)", test_a0_N0_001, "(0, 0.001)",x_axis=range(0,len(test_a0_N0_25)), x_label="Epochs", y_label="Loss")
    
    return


def investigate_lr_nparam(ansatz, test_train):
    sns.set_style("darkgrid")
    plt.rcParams.update({'font.size':14})
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams["figure.figsize"] = (13.1,8.65)   #(13.1,7.4)
    #plt.rcParams['xtick.labelsize'] = 20
    #plt.rcParams['ytick.labelsize'] = 20
    rates = [0.1, 0.01, 0.001, 0.0001]
    Ns = [10,14,18,22, 26, 30]

    colors = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:olive"]
    ticks = ["*","s",".","D"]

    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    start=16

    for N in range(len(Ns)):
        for rate in range(len(rates)):
            filename_loss = "Results/saved_data/iris/ansatz_"+str(ansatz)+"/lr_params/"+test_train+"_lr%s_n%s.npy" %(rates[rate], Ns[N])
            #print(filename_loss)
            loss = np.fromfile(filename_loss)
            loss = loss[start:len(loss)+start]
        
            iterations = np.arange(len(loss))
            ax2.plot(iterations,loss,color=colors[N],marker=ticks[rate],label=r"n=%s;$\gamma$=%s" %(Ns[N],rates[rate]))

    ax2.legend(prop={'size':30})
    ax2.set_xlabel("Epochs",fontsize=16)
    ax2.set_ylabel("Loss",fontsize=16)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.92, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return


def optimal_run(folder):  
    a0_train_loss=np.load("Results/saved_data/"+folder+"/ansatz_0/trainloss_optimal.npy")
    a0_test_loss=np.load("Results/saved_data/"+folder+"/ansatz_0/testloss_optimal.npy")
    a0_train_acc=np.load("Results/saved_data/"+folder+"/ansatz_0/trainacc_optimal.npy")
    a0_test_acc=np.load("Results/saved_data/"+folder+"/ansatz_0/testacc_optimal.npy")
    
    a1_train_loss=np.load("Results/saved_data/"+folder+"/ansatz_1/trainloss_optimal.npy")
    a1_test_loss=np.load("Results/saved_data/"+folder+"/ansatz_1/testloss_optimal.npy")
    a1_train_acc=np.load("Results/saved_data/"+folder+"/ansatz_1/trainacc_optimal.npy")
    a1_test_acc=np.load("Results/saved_data/"+folder+"/ansatz_1/testacc_optimal.npy")
    
    plotter(a0_train_loss, "Train loss",  a0_test_loss, "Validation loss", a0_train_acc, "Train Accuracy", a0_test_acc, "Validation accuracy",x_axis=range(0,len(a0_train_loss)), x_label="Epochs", y_label="Loss & Accuracy")
    plotter(a1_train_loss, "Train loss",  a1_test_loss, "Validation loss", a1_train_acc, "Train Accuracy", a1_test_acc, "Validation accuracy",x_axis=range(0,len(a1_train_loss)), x_label="Epochs", y_label="Loss & Accuracy")
    
    a0_best_acc=np.nanmax(a0_test_acc)
    a0_acc_epoch=np.where(a0_test_acc==np.nanmax(a0_test_acc))
    a0_best_loss=np.nanmin(a0_test_loss)
    a0_loss_epoch=np.where(a0_test_loss==np.nanmin(a0_test_loss))
    
    a1_best_acc=np.nanmax(a1_test_acc)
    a1_acc_epoch=np.where(a1_test_acc==np.nanmax(a1_test_acc))
    a1_best_loss=np.nanmin(a1_test_loss)
    a1_loss_epoch=np.where(a1_test_loss==np.nanmin(a1_test_loss))
    
    print("Ansats 1:")
    print(f"Best accuracy      : {a0_best_acc}")
    print(f"Best accuracy epoch:{a0_acc_epoch}")
    print(f"Best loss          : {a0_best_loss}")
    print(f"Best loss epoch    : {a0_loss_epoch}")

    print("Ansats 2:")
    print(f"Best accuracy      : {a1_best_acc}")
    print(f"Best accuracy epoch:{a1_acc_epoch}")
    print(f"Best loss          : {a1_best_loss}")
    print(f"Best loss epoch    : {a1_loss_epoch}")
        
    return


#plott_distribution()
#investigate_lr_nparam(0, "test")
optimal_run("iris")
