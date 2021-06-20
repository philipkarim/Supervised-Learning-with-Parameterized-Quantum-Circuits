#This import is just because of some duplicate of mpi or armadillo on the computer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Importing packages 
import numpy as np
import matplotlib.pyplot as plt

#Importing pytorch packages
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

#Importing qiskit
import qiskit as qk
from qiskit.visualization import *
from quantumnet import *


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
x = iris.data
y = iris.target
idx = np.where(y < 2) # we only take the first two targets.

x = x[idx,:]
y = y[idx]

p = 2 #number of features
p_classic=1 #Classical measurements
data_register = qk.QuantumRegister(p)
classical_register = qk.ClassicalRegister(p_classic)

circuit = qk.QuantumCircuit(data_register, classical_register)

sample = np.random.uniform(size=p)
target = np.random.uniform(size=1)

for feature_idx in range(p):
    circuit.h(data_register[feature_idx])
    circuit.rz(2*np.pi*sample[feature_idx],data_register[feature_idx])

print(circuit)
