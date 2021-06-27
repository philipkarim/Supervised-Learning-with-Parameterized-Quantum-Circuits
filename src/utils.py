# Common imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

random.seed(2021)

def accuracy_score(y, y_pred):
    """
    Computes the accuracy score
    """
    y=np.ravel(y)
    y_pred=np.ravel(y_pred)
    numerator=np.sum(y == y_pred)

    return numerator/len(y)

def hard_labels(y_array, treshold):
    """
    Rewriting the soft predictions in probabillity space
    into hard predictions, of 0 or 1
    """
    for i in range(len(y_array)):
        y_array[i]=np.where(y_array[i]>treshold, 1, 0)
    
    return y_array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getDistribution(type, stop, n):
    if type=="U":
        return np.random.uniform(0.,stop,size=n)
    elif type=="N":
        return np.random.normal(stop/2,stop,size=n)
    else:
        print("Choose U for uniform distribution or N for normal distribution. \n Shutting down")
        quit()

def data_path(DATA_ID, dat_id):
    return os.path.join(DATA_ID, dat_id)


def plotter(*args, x_axis,x_label, y_label):
    """
    Just a function to plot functions.

    Args: *args:    arguments passed as x1 data, y1 data, 
                    label 1, x2 data, y2 data, label 2...
          x_label:  Name of x axis(string)
          y_label:  Name of y axis(string)
    """
    sns.set_style("darkgrid")
    
    if len(args)>1:
        for i in range(0, int(len(args)),2):
            plt.plot(x_axis, args[i], label=args[i+1])
            plt.legend()
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
        plt.ylim(0.35,0.67)
    else:
        plt.plot(x_axis, args[0])
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    plt.show()

    return