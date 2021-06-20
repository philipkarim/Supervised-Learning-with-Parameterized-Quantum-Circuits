import qiskit
from qiskit import transpile, assemble
import numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch

#This functions has every class regarding the neural networks
class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        """
        Class that creates the quantum circuit, outline of the class is
        based on the framework in the qiskit textbook

        Args:
            n_qubits: Amount of qubits in circuit

        """
        #creating the circuit
        self.QCircuit = qiskit.QuantumCircuit(n_qubits)
        
        #Amount of qubits
        all_qubits=list(range(0,n_qubits))

        #Variational parameter
        self.theta = qiskit.circuit.Parameter('theta')
        
        #quantum gates
        self.QCircuit.h(all_qubits)
        self.QCircuit.barrier()
        self.QCircuit.ry(self.theta, all_qubits)
        self.QCircuit.measure_all()

        self.backend = backend
        self.shots = shots
    
    def run(self, parameter_theta):
        """
        Running the simulation simulation
        Args: 
            thetas: Number of variational parameters
        
        Returns:
            np.array([expectation])  : Expectation value
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

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

