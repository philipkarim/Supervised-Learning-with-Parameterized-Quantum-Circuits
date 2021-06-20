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
import qiskit
from qiskit.visualization import *
from quantumnet import *

#Just testing the circuit
simulator = qiskit.Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(1, simulator, 100)
#print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
#print(circuit.QCircuit)


#Preparing the data
n_samples_train = 100
n_samples_test =50

X_train = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
X_test = datasets.MNIST(root='./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))


# Want to deal with the numbers 1 and 7 
idx_train = np.append(np.where(X_train.targets == 0)[0][:n_samples_train], np.where(X_train.targets == 1)[0][:n_samples_train])
idx_test = np.append(np.where(X_test.targets == 0)[0][:n_samples_test], np.where(X_test.targets == 1)[0][:n_samples_test])

#Setting the 1's and 7's as train and test data
X_train.data = X_train.data[idx_train]
X_train.targets = X_train.targets[idx_train]
X_test.data = X_test.data[idx_test]
X_test.targets = X_test.targets[idx_test]

#Preparing the dataloader
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

#Samples to show from the data
n_data = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_data, figsize=(10, 3))

while n_data > 0:
    images, targets = data_iter.__next__()

    axes[n_data - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
    axes[n_data - 1].set_xticks([])
    axes[n_data - 1].set_yticks([])
    axes[n_data - 1].set_title("Labeled: {}".format(targets.item()))
    
    n_data -= 1
#plt.show()


#Creating the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)



#Training the network
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss_func = nn.CrossEntropyLoss()
#loss_func = nn.BCEWithLogitsLoss()
#loss_func = nn.MSELoss()
loss_func = nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))

plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg Log Likelihood Loss')
plt.show()

#Testing the network
model.eval()
with torch.no_grad():
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100))

n_data_train= 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_data_train, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_data_train:
            break
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 

        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(pred.item()))
        
        count += 1
    plt.show()



#Steps:
#Rewrite the program
#Try with sgd or gradient descent instead of adam to see if the accuracy reduces
#Try with the ansatz from project 2