import qiskit as qk
import numpy as np


class QML:
    def __init__(self, ansatz, n_qubits, n_cbits, n_parameters, backend="qasm_simulator", shots=1024):
        """
        Class that creates a parameterized quantum circuit

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

        #print(self.qc_enc)

        return self.qc_enc
    
    def predict(self, designX, params):
        #Splits up the design matrix
        predictions_array=np.zeros(designX.shape[0])

        for samp in range(designX.shape[0]):
            self.create_qcircuit(designX[samp], params)
            predictions_array[samp]=self.run()

        return predictions_array
    
    def run(self):
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
