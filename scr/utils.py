# Common imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def data_path(DATA_ID, dat_id):
    return os.path.join(DATA_ID, dat_id)

sns.set_style("darkgrid")

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Violet]
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

def plotter(*args, x_label, y_label):
    """
    Just a function to plot functions.

    Args: *args:    arguments passed as x1 data, y1 data, 
                    label 1, x2 data, y2 data, label 2...
          x_label:  Name of x axis(string)
          y_label:  Name of y axis(string)
    """
    if args>3:
        for i in range(len(args)/3):
            plt.plot(args[i], args[i+1], label=args[i+2])
        plt.legend()

    else:
        plt.plot(args[0], args[1], label=args[2])
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    plt.show()

    return