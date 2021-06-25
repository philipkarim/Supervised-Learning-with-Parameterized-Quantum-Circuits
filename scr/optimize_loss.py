import numpy as np

class optimize:
    def __init__(self, learning_rate, circuit):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
        """

        self.learning_rate=learning_rate
        self.circuit=circuit

    def gradient_descent(self, params, predicted, target, samples):
        update=self.learning_rate *self.gradient_of_loss(params, predicted, target, samples)
        #print(update)
        params+= update

        return params

    def cross_entropy(self, preds, targets, classes=2, epsilon=1e-12):
        """
        Computes cross entropy between the true labels and the predictions
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        #Creates matrixes to use one hot encoded labels
        distribution_preds=np.zeros((len(preds), classes))
        distribution_target=np.zeros((len(targets), classes))

        #Just rewriting the predictions and labels
        for i in range(len(preds)):
            distribution_preds[i][0]=1-preds[i]
            distribution_preds[i][1]=preds[i]
            
            if targets[i]==0:
                distribution_target[i][0]=1
            elif targets[i]==1:
                distribution_target[i][1]=1

        distribution_preds = np.clip(distribution_preds, epsilon, 1. - epsilon)
        n_samples = len(preds)
        loss = -np.sum(distribution_target*np.log(distribution_preds+1e-9))/n_samples
        return loss

    def binary_cross_entropy(self, preds, targets, classes=2, epsilon=1e-12):
        """
        Computes binary cross entropy between the true labels and the predictions
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        sum=0
        n_samples=len(preds)
        for index in range(n_samples):
            sum+=targets[index]*np.log(preds[index])+(1-targets[index])*np.log(1-preds[index])

        return -sum/n_samples


        distribution_preds = np.clip(distribution_preds, epsilon, 1. - epsilon)
        n_samples = len(preds)
        loss = -np.sum(distribution_target*np.log(distribution_preds+1e-9))/n_samples
        return loss

    def parameter_shift(self, sample, theta_array, theta_index):
        theta_left_shift=theta_array.copy()
        theta_right_shift=theta_array.copy()
        #Since the cirquits are normalised the shift is 0.25 which represents pi/2
        theta_right_shift[theta_index]+=0.25
        theta_left_shift[theta_index]-=0.25
        
        """
        print("_____this")
        print(theta_index)
        print(theta_array)
        print(theta_right_shift)
        print(theta_left_shift)
        print("________this")
        """

        #This is quiet weird, even though the parameter shift are different, 
        #the prediction is still the same, is this due to the quantum mechanics?

        #print(theta_right_shift)
        #print(theta_left_shift)
        pred_right_shift=self.circuit.predict(np.array([sample]),theta_right_shift)
        pred_left_shift=self.circuit.predict(np.array([sample]),theta_left_shift)
        #print(pred_right_shift)
        #print(pred_left_shift)
        theta_grad=(pred_right_shift[0]-pred_left_shift[0])/2
        #print(theta_grad)
        return theta_grad

    def gradient_of_loss(self, thetas, predicted, target, samples):
        gradients=np.zeros(len(thetas))
        eps=1E-8
        for thet in range(len(thetas)):
            sum=0
            for i in range(len(predicted)):
                #Really unsure about this one, okay maybe not,
                #this works i think
                #print(samples[i], thetas, thet)
                grad_theta=self.parameter_shift(samples[i], thetas, thet)
                #print(grad_theta)
                #print(thet, grad_theta)
                #print(grad_theta)
                deno=(predicted[i]+eps)*(1-predicted[i]-eps)
                sum+=grad_theta*abs(predicted[i]-target[i])/deno
            gradients[thet]=sum
        
        return gradients