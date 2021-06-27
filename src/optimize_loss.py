import numpy as np

class optimize:
    def __init__(self, learning_rate, circuit):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
            circuit:        Quantum circuit that is used
        """

        self.learning_rate=learning_rate
        self.circuit=circuit

    def gradient_descent(self, params, predicted, target, samples):
        """
        Gradient descent function.
        
        Args:
                params:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)
        """
        update=self.learning_rate *self.gradient_of_loss(params, predicted, target, samples)
        params-= update

        return params

    def cross_entropy(self, preds, targets, classes=2, epsilon=1e-12):
        """
        Computes cross entropy between the true labels and the predictions
        by using distributions. (Not used, binary cross entropy function beneath)
        
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

    def binary_cross_entropy(self, preds, targets):
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


    def parameter_shift(self, sample, theta_array, theta_index):
        """
        Parameter shift funtion, that finds the derivative of a
        certain variational parameter, by evaluating the cirquit
        shiftet pi/2 to the left and right.

        Args:
                sample:     Sample that will be optimized with respect 
                            of(list or array)
                theta_array:Variational parameters(list or array)
                theta_index:Index of the parameter that we want the 
                            gradient of

        Returns: optimized variational value
        """
        #Just copying the parameter arrays
        theta_left_shift=theta_array.copy()
        theta_right_shift=theta_array.copy()
        
        #Since the cirquits are normalised the shift is 0.25 which represents pi/2
        theta_right_shift[theta_index]+=0.25
        theta_left_shift[theta_index]-=0.25

        #Predicting with the shifted parameters
        pred_right_shift=self.circuit.predict(np.array([sample]),theta_right_shift)
        pred_left_shift=self.circuit.predict(np.array([sample]),theta_left_shift)
        
        #Computes the gradients
        theta_grad=(pred_right_shift[0]-pred_left_shift[0])/2

        return theta_grad

    def gradient_of_loss(self, thetas, predicted, target, samples):
        """
        Finds the gradient used in gradient descent.

        Args:
                thetas:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)

        Returns:            Gradient 
        """

        gradients=np.zeros(len(thetas))
        eps=1E-8
        #Looping through variational parameters
        for thet in range(len(thetas)):
            sum=0
            #Looping through samples
            for i in range(len(predicted)):
                grad_theta=self.parameter_shift(samples[i], thetas, thet)
                deno=(predicted[i]+eps)*(1-predicted[i]-eps)
                sum+=grad_theta*(predicted[i]-target[i])/deno

            gradients[thet]=sum
        
        return gradients