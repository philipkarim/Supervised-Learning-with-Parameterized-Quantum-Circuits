# Supervised Learning with Parameterized Quantum Circuits to Classify Samples from the Iris Dataset

## Code structure
### Main program
All parameters are defined in `main.py`, the training and testing also happens in `main.py`. The most fundamental parts of the code are `PQC.py` and `optimize_loss.py`. The former is a quantum cirquit class handling everything to do with the quantum cirquits and the simulations of them, while the latter is handling the loss and optimization of the variational parameters by using gradient descent. `utils.py` consists of usefull functions, that are used various places in the scripts.

### Analytics
The code used for plotting and analyzing the results can be found in `analyzer.py`.

## Running the project:
The requirements used to run the project can be found in `requirements.txt`. In addition, when running the functions `investigate_distribution()` and `investigate_lr_params()` in `main.py` a unix based system is required due to the utillization of the fork() command. The project can still be ran with the `train.py()` function.
