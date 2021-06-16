import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

"""
n = 8
n_q = n
n_b = n
qc_output = QuantumCircuit(n_q,n_b)

for j in range(n):
    qc_output.measure(j,j)



qc_output.draw()
print(qc_output)
sim = Aer.get_backend('qasm_simulator')  # this is the simulator we'll use
qobj = assemble(qc_output)  # this turns the circuit into an object our backend can run
result = sim.run(qobj).result()  # we run the experiment and get the result from that experiment
#from the results, we get a dictionary containing the number of times (counts)
#each result appeared
counts = result.get_counts()
#and display it on a histogram
#plot_histogram(counts)
#plt.show()
#print(qc_output)

qc_encode = QuantumCircuit(n)
qc_encode.x(7)
qc_encode.draw()

#print(qc_output)
qc=qc_output.compose(qc_encode, front=True, inplace=True)
print(qc_output)

qobj = assemble(qc_output)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)
plt.show()

#Parameterized r phi gate

qc = QuantumCircuit(1)
qc.rz(np.pi/4, 0)
qc.draw()
print(qc)
"""

#Multiple qubit entanglement

from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector

qc = QuantumCircuit(3)
# Apply H-gate to each qubit:
for qubit in range(3):
    qc.h(qubit)
# See the circuit:
#print(qc)
#Saves the circuit
#qc.draw(output='mpl', filename='my_circuit_test.png')

# Let's see the result
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
#print(final_state)

#the whole circuit can be written as a matrix
qc2 = QuantumCircuit(2)
qc2.h(0)
qc2.x(1)
#print(qc2)
#The unitary simulator gives the finished circuit
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc2)
unitary = usim.run(qobj).result().get_unitary()
#print(unitary)

#the whole circuit can be written as a matrix
qc3 = QuantumCircuit(2)
qc3.x(0)
qc3.z(0)
qc3.h(0)
print(qc3)
#The unitary simulator gives the finished circuit
usim = Aer.get_backend('unitary_simulator')
qobj = assemble(qc3)
unitary = usim.run(qobj).result().get_unitary()
print(unitary)