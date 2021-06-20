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

qc_enc = qk.QuantumCircuit(data_register, classical_register)

sample = np.random.uniform(size=p)
target = np.random.uniform(size=1)

sample=[4,5]

for feature_idx in range(p):
    qc_enc.h(data_register[feature_idx])
    #rz(phi, which qubit)
    qc_enc.rz(2*np.pi*sample[feature_idx],data_register[feature_idx])

print(qc_enc)

"""
#Lets try with the first sample of the iris dataset:
iris_features=len(x[0][0])
p_classic=1 #Classical measurements
data_register_iris = qk.QuantumRegister(iris_features)
circuit_enc = qk.QuantumCircuit(data_register_iris, classical_register)

first_sample=x[0][0]

for feature_idx in range(len(first_sample)):
    circuit_enc.h(data_register_iris[feature_idx])
    #rz(phi, which qubit)
    circuit_enc.rz(2*np.pi*first_sample[feature_idx],data_register_iris[feature_idx])

#print(first_sample)
#print(circuit_enc)
"""

#Parameterized the gates connected to the first part
n_params = 4
theta = 2*np.pi*np.random.uniform(size=n_params)
theta=[0,1,2,3]
circuit_ansatz_example = qk.QuantumCircuit(data_register, classical_register)

circuit_ansatz_example.ry(theta[0], data_register[0])
circuit_ansatz_example.ry(theta[1], data_register[1])

circuit_ansatz_example.cx(data_register[0], data_register[1])

circuit_ansatz_example.ry(theta[2], data_register[0])
circuit_ansatz_example.ry(theta[3], data_register[1])

circuit_ansatz_example.cx(data_register[0], data_register[1])

print(circuit_ansatz_example)

#then it is time for the measurement part
#First we gotta merge the encoding circuit and the ansats circuit together 

#qc_composed=circuit_ansatz_example.compose(circuit_enc, front=True, inplace=True)
qc_enc.compose(circuit_ansatz_example, front=False, inplace=True)
#qc_composed=circuit_enc+circuit_ansatz_example
#qc_enc2.copy(name=qc_enc)

print(qc_enc)

"""
qc_composed.measure(data_register[-1],classical_register[0])
shots=1000

job = qk.execute(circuit,
                backend=qk.Aer.get_backend('qasm_simulator'),
                shots=shots,
                seed_simulator=42
                )
results = job.result()
results = results.get_counts(circuit)

prediction = 0
for key,value in results.items():
    if key == '1':
        prediction += value
prediction/=shots
print('Prediction:',prediction,'Target:',target[0])
"""