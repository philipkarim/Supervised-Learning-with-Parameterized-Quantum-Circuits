import numpy as np

class optimize:
    def __init__(self, learning_rate):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
        """

        self.learning_rate=learning_rate

    def gradient_descent(self, derivative, initial_value, learn_rate, n_iter):
        thetas = initial_value
        for _ in range(n_iter):
            diff = -learn_rate * derivative(thetas)
            thetas += diff

            #alpha += lambda*energyDerivative

        return thetas

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

    def parameter_shift(self, circuit, sample, theta_array, theta_index):
        theta_left_shift=theta_array
        theta_right_shift=theta_array
        #Since the cirquits are normalised the shift is 0.25 which represents pi/2
        theta_right_shift[theta_index]+=0.25
        theta_left_shift[theta_index]-=0.25

        pred_right_shift=circuit.predict(sample,theta_right_shift)
        pred_left_shift=circuit.predict(sample,theta_left_shift)

        theta_grad=(pred_right_shift[0]-pred_left_shift[0])/2

        return theta_grad

    def gradient_of_loss(self, predicted, target, samples, thetas):
        gradients=np.zeros(len(thetas))

        for thet in range(len(thetas)):
            sum=0
            for i in range(len(predicted)):
                #Really unsure about this one, okay maybe not,
                #this works i think
                grad_theta=parameter_shift(samples[i], thetas, thet)
                deno=predicted[i]*(1-predicted[i])
                sum+=grad_theta*abs(predicted[i]-target[i])/deno
            gradients[thet]=sum
        
        return gradients