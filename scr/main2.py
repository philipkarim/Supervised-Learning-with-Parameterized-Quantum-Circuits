"""

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

qc_enc.measure(data_register[-1],classical_register[0])

shots=1000

job = qk.execute(qc_enc,
                backend=qk.Aer.get_backend('qasm_simulator'),
                shots=shots,
                seed_simulator=42
                )
results = job.result()
results = results.get_counts(qc_enc)

prediction = 0
for key,value in results.items():
    print(key, value)
    if key == '1':
        prediction += value
prediction/=shots
print('Prediction:',prediction,'Target:',target[0])
print(qc_enc)

"""

#This import is just because of some duplicate of mpi or armadillo on the computer
import os

from qiskit.circuit import parameter
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
from sklearn.model_selection import train_test_split

#Handling the data
iris = datasets.load_iris()
x = iris.data
y = iris.target
idx = np.where(y < 2) # we only take the first two targets.

x = x[idx,:]
x=np.squeeze(x, axis=0)
y = y[idx]

#X, X_test, y, y_test = train_test_split(X,y,test_size=0.92,stratify=y)
X_train,X_test, y_train, y_test = train_test_split(x,y,test_size=0.8)
#stratisfy=y?
#print('train dataset shape', X_train.shape)
#print('val dataset shape', X_val.shape)
#print('test dataset shape', X_test.shape)
"""
p = x.shape[1] #number of features

p_classic=1 #Classical measurements

data_register = qk.QuantumRegister(p)
classical_register = qk.ClassicalRegister(p_classic)

qc_enc = qk.QuantumCircuit(data_register, classical_register)

sample = np.random.uniform(size=p)
target = np.random.uniform(size=1)

for feature_idx in range(p):
    qc_enc.h(data_register[feature_idx])
    #rz(phi, which qubit)
    qc_enc.rz(2*np.pi*sample[feature_idx],data_register[feature_idx])

print(qc_enc)


#Parameterized the gates connected to the first part
#figure 5 ansatz
n_params = p*2
theta = 2*np.pi*np.random.uniform(size=n_params)
qc_ansatz_1 = qk.QuantumCircuit(data_register, classical_register)

for rot_y in range(p):
    qc_ansatz_1.ry(theta[rot_y], data_register[rot_y])
    if rot_y!=0:
        for con_x in range(rot_y):
            qc_ansatz_1.cx(data_register[con_x], data_register[rot_y])
print(qc_ansatz_1)

#could add some more to the circuit but dont know how one should measure the states

#then it is time for the measurement part
#First we gotta merge the encoding circuit and the ansats circuit together 
qc_enc.compose(qc_ansatz_1, front=False, inplace=True)
qc_enc.measure(data_register[-1],classical_register[0])

#Run the simulator predict the last qubit
def run_qc(circuit, shots=1024, backend_sim='qasm_simulator'):
    job = qk.execute(circuit,
                    backend=qk.Aer.get_backend(backend_sim),
                    shots=shots,
                    seed_simulator=10
                    )
    results = job.result()
    results = results.get_counts(circuit)

    prediction = 0
    for key,value in results.items():
        #print(key, value)
        if key == '1':
            prediction += value
    prediction/=shots

    return prediction

prediction=run_qc(qc_enc)

print('Prediction:',prediction,'Target:',target[0])
print(qc_enc)

"""

#This functions has every class regarding the neural networks
class QML:
    def __init__(self, ansatz, n_qubits, n_cbits, n_parameters, backend="qasm_simulator", shots=1024):
        """
        Class that creates the quantum circuit, outline of the class is
        based on the framework in the qiskit textbook

        Args:
            ansatz: 0=ry along all qubits, in addition to cx between all gates as follows:
                           ┌────────────┐
                    q0_0: ─┤ RY(\theta0)├───■────■─────────■────────────
                           ├────────────┤ ┌─┴─┐  │         │
                    q0_1: ─┤ RY(\theta1)├─┤ X ├──┼────■────┼────■───────
                           ├────────────┤ └───┘┌─┴─┐┌─┴─┐  │    │
                    q0_2: ─┤ RY(\theta0)├──────┤ X ├┤ X ├──┼────┼────■──
                          ┌┴────────────┴┐     └───┘└───┘┌─┴─┐┌─┴─┐┌─┴─┐
                    q0_3: ┤ RY(\theta0)  ├───────────────┤ X ├┤ X ├┤ X ├
                          └──────────────┘               └───┘└───┘└───┘
                    c0: 1/══════════════════════════════════════════════

                    1= 
            
            n_qubits: How many qubits in the quantum register
            n_cbits:  How many classical bits to save the measures on?
            backend: What kind of simulator to use, qasm_simulator is preferred
            shots:   How many times the qunautm circuit is to be repeated
        """

        self.ansatz=ansatz
        self.n_qubits=n_qubits
        self.n_cbits=n_cbits
        self.backend = backend
        self.shots = shots
        self.n_parameters=n_parameters

        #Variational parameter
        #self.theta = qk.circuit.Parameter('theta')

        #self.theta=np.random.uniform(size=)

        #Registers
        self.data_register = qk.QuantumRegister(n_qubits)
        self.classical_register = qk.ClassicalRegister(n_cbits)

    def make_encoder_cir(self, sample):
        """
        Creates the encoding circuit
        
        Args:
            n_qubits: How many qubits in the quantum register
            n_cbits:  How many classical bits to save the measures on?

        Returns:
            Encoding circuit containing the classical data as a quantum circuit
        """

        self.qc_enc = qk.QuantumCircuit(self.data_register, self.classical_register)

        for feature_idx in range(self.n_qubits):
            self.qc_enc.h(self.data_register[feature_idx])
            self.qc_enc.rz(2*np.pi*sample[feature_idx],self.data_register[feature_idx])
        
        return self.qc_enc

    def make_param_cir(self, thetas):
        """
        Creates the parameterized circuit

        Args: 
            x_design: Design matrix
            thetas: Number of variational parameters
        
        Returns: 
            Predicted value (float))
        """
        self.thetas=thetas
        self.qc_ansatz = qk.QuantumCircuit(self.data_register, self.classical_register)
        
        ansatz_parts=self.n_parameters//self.n_qubits
        reminder_gates=self.n_parameters%self.n_qubits

        if self.ansatz==0:
            #Creating the ansatz circuit:
            if reminder_gates!=0:
                blocks=ansatz_parts+1
            else:
                blocks=ansatz_parts

            for block in range(blocks):
                for rot_y in range(self.n_qubits):
                    if rot_y+4*block<self.n_parameters:
                        self.qc_ansatz.ry(self.thetas[rot_y+4*block], self.data_register[rot_y])
                    if rot_y!=0:
                        for con_x in range(rot_y):
                            self.qc_ansatz.cx(self.data_register[con_x], self.data_register[rot_y])

            """
            #Copies the ansatz multiple times to ensure that the wanted number of parameters is used:
            if ansatz_parts>1:
                for ansatz in range(ansatz_parts-1):
                    self.qc_ansatz.compose(self.qc_ansatz, front=False, inplace=True)

            #Adds the extra reminder gates
            for rot_y_reminder in range(reminder_gates):
                self.qc_ansatz.ry(self.thetas[rot_y_reminder], self.data_register[rot_y_reminder])
                if rot_y_reminder!=0:
                    for con_x_reminder in range(rot_y_reminder):
                        self.qc_ansatz.cx(self.data_register[con_x_reminder], self.data_register[rot_y_reminder])
            """
        return self.qc_ansatz
    
    def create_qcircuit(self, sample, parameters):
        #Assigns values to the theta parameter defined in init
        #self.theta.assign(self.theta,parameters)

        #for thet in range(len(parameters)):
        #    self.theta[thet]=parameters[thet]

        self.make_encoder_cir(sample)
        self.make_param_cir(parameters)

        self.qc_enc.compose(self.qc_ansatz, front=False, inplace=True)
        self.qc_enc.measure(self.data_register[-1],self.classical_register[0])

        print(self.qc_enc)

        return self.qc_enc
    
    def predict(self):
        job = qk.execute(self.qc_enc,
                        backend=qk.Aer.get_backend(self.backend),
                        shots=self.shots,
                        seed_simulator=10
                        )
        results = job.result()
        results = results.get_counts(self.qc_enc)

        prediction = 0
        for key,value in results.items():
            #print(key, value)
            if key == '1':
                prediction += value
        prediction/=self.shots

        return prediction
#    def predict(self, X_design, thetas):
        """
        Running the simulation simulation
        Args: 
            x_design: Design matrix
            thetas: Number of variational parameters
        
        Returns: 
            Predicted value (float))
        """
"""
        #Standard procedure when running the circuit
        theta_dict_list=[]
        for i in range(0,len(parameter_theta)):
            theta_dict={}
            theta_dict[self.theta]=parameter_theta[i]
            theta_dict_list.append(theta_dict)
        #Or maybe this way? [{1:1}, {2:2}] instead of [{1:1, 2:2}]?
        #theta_dict=[{self.theta: theta_val} for theta_val in parameter_theta]

        transpiled_qc = transpile(self.QCircuit, self.backend)
        quantumobject = assemble(transpiled_qc, shots=self.shots, parameter_binds = theta_dict_list)
        job = self.backend.run(quantumobject)
        #Gets the counts
        result = job.result().get_counts()
        
        #Saves values and states as an array
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
                
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expec = np.sum(states * probabilities)
        
        return np.array([expec])
"""


#First expand the example now to iris dataset
#Rewrite to iris dataset, place everything inside a function, where the ansatz can be chosen as a parameter
#Make a design matrix X which is basicly all flower samples under eachother?

#First expand the example now to iris dataset
#Rewrite to iris dataset, place everything inside a function, where the ansatz can be chosen as a parameter
#Make a design matrix X which is basicly all flower samples under eachother?

#Sende inn et og et sample og lagre the prediction for alle sammen, etter den er ferdig så kan dataen
#settes sammen til en design matrix med sine predicted values

"""
       ┌────────────┐
q0_0: ─┤ RY(\theta0)├───■────■─────────■────────────
       ├────────────┤ ┌─┴─┐  │         │
q0_1: ─┤ RY(\theta1)├─┤ X ├──┼────■────┼────■───────
       ├────────────┤ └───┘┌─┴─┐┌─┴─┐  │    │
q0_2: ─┤ RY(\theta0)├──────┤ X ├┤ X ├──┼────┼────■──
      ┌┴────────────┴┐     └───┘└───┘┌─┴─┐┌─┴─┐┌─┴─┐
q0_3: ┤ RY(\theta0)  ├───────────────┤ X ├┤ X ├┤ X ├
      └──────────────┘               └───┘└───┘└───┘
c0: 1/══════════════════════════════════════════════
"""

n_parameters=10
qc=QML(0, x.shape[1],1,n_parameters, backend="qasm_simulator", shots=1024)
thetas = np.random.uniform(size=n_parameters)
qc.create_qcircuit(x[0],thetas)
prediction=qc.predict()
print(prediction)
#Next steps:
"""
-Fixt the looping over the design matrix inserting one and one sample
-Try training the thing
-Try adding epochs and such to see how much the loss/accuracy is
-Maybe spread out the ansatz making it look better
-Add another ansatz, cool one in lin 6 bookmark
-try both ansatzes on breast cancer dataset
"""