import qiskit as qk
import numpy as np
from qiskit.circuit.library.standard_gates import XGate

class QML:
    def __init__(self, ansatz, n_qubits, n_cbits, n_parameters, backend="qasm_simulator", shots=1024):
        """
        Class that creates a parameterized quantum circuit

        Args:
            ansatz: 0=ry along all qubits, in addition to cx between pairwise gates as follows
            (Parametervalues in the circuits depends on the input parameters):
                          ┌───────┐      ┌────────┐                      ┌─────────┐     
                    q0_0: ┤ RY(0) ├───■──┤ RY(8π) ├────────────────■─────┤ RY(16π) ├──■──
                          ├───────┴┐┌─┴─┐└────────┘┌─────────┐   ┌─┴─┐   ├─────────┤  │  
                    q0_1: ┤ RY(2π) ├┤ X ├────■─────┤ RY(10π) ├───┤ X ├───┤ RY(18π) ├──■──
                          ├────────┤└───┘  ┌─┴─┐   └─────────┘┌──┴───┴──┐└─────────┘  │  
                    q0_2: ┤ RY(4π) ├───────┤ X ├────────■─────┤ RY(12π) ├─────────────■──
                          ├────────┤       └───┘      ┌─┴─┐   ├─────────┤           ┌─┴─┐
                    q0_3: ┤ RY(6π) ├──────────────────┤ X ├───┤ RY(14π) ├───────────┤ X ├
                          └────────┘                  └───┘   └─────────┘           └───┘
                    c0: 1/═══════════════════════════════════════════════════════════════



            ansatz: 1=ry along all qubits, in addition to cx between all gates as follows:
                           ┌────────────┐
                    q0_0: ─┤ RY(\theta0)├───■────■─────────■────────────
                           ├────────────┤ ┌─┴─┐  │         │
                    q0_1: ─┤ RY(\theta1)├─┤ X ├──┼────■────┼────■───────
                           ├────────────┤ └───┘┌─┴─┐┌─┴─┐  │    │
                    q0_2: ─┤ RY(\theta2)├──────┤ X ├┤ X ├──┼────┼────■──
                          ┌┴────────────┴┐     └───┘└───┘┌─┴─┐┌─┴─┐┌─┴─┐
                    q0_3: ┤ RY(\theta3)  ├───────────────┤ X ├┤ X ├┤ X ├
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
            #Checks how many "blocks of repetitions goes in the circuit"
            if reminder_gates!=0:
                blocks=ansatz_parts+1
            else:
                blocks=ansatz_parts

            #Creating the ansatz circuit:
            tot_gates=0
            for block in range(blocks):
                for rot_y in range(self.n_qubits):
                    if rot_y+self.n_qubits*block<self.n_parameters:
                        self.qc_ansatz.ry(2*np.pi*self.thetas[rot_y+self.n_qubits*block], self.data_register[rot_y])
                        tot_gates+=1
                    if rot_y!=0 and tot_gates<len(self.thetas)-reminder_gates-1:
                        self.qc_ansatz.cx(self.data_register[rot_y-1], self.data_register[rot_y])
            
            #Entangling the qubits before measuring
            c3h_gate = XGate().control(3)
            self.qc_ansatz.append(c3h_gate, self.data_register)

        elif self.ansatz==1:
            #Creating the ansatz circuit:
            if reminder_gates!=0:
                blocks=ansatz_parts+1
            else:
                blocks=ansatz_parts

            for block in range(blocks):
                for rot_y in range(self.n_qubits):
                    if rot_y+self.n_qubits*block<self.n_parameters:
                        self.qc_ansatz.ry(2*np.pi*self.thetas[rot_y+4*block], self.data_register[rot_y])
                    if rot_y!=0:
                        for con_x in range(rot_y):
                            self.qc_ansatz.cx(self.data_register[con_x], self.data_register[rot_y])

        else:
            print("Chosen ansatz, is not available, set ansatz to 0 or 1.\n Terminating program")
            quit()

        return self.qc_ansatz
    
    def create_qcircuit(self, sample, parameters):
        """
        Puts the cirquit together and assigns parameter values to each
        of the gates

        Args:   
                sample:     Values of the samples that are going 
                            into the cirquit(list or array)
                parameters: variational parameters (list or array)
        """

        self.make_encoder_cir(sample)
        self.make_param_cir(parameters)
        self.qc_enc.compose(self.qc_ansatz, front=False, inplace=True)
        self.qc_enc.measure(self.data_register[-1],self.classical_register[0])

        #print(self.qc_enc)

        return self.qc_enc

    def predict(self, designX, params):
        """
        Predicts the output of the cirquit.

        Args:
                designX:    Design matrix(Matrix, array or list)
                params:     Variational parameters
        
        Returns:List of prediction values
        """
        predictions_list=[]

        #Predicting the samples and re
        for samp in range(designX.shape[0]):
            self.create_qcircuit(designX[samp], params)
            predictions_list.append((self.run()))
        
        return predictions_list
    
    def run(self):
        """
        Function that runs the quantum simulator
        """
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
