import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self):
         super(CNN,self).__init__()

         #defining the convultional layers (if kernal size is 3 and padding is 1, then
         #the length will extend by 3 max - wait actually idk)
         self.conv1 = nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3,padding=1)
         self.conv2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding=1)

         #define the fully connected layers
         self.fc1 = nn.Linear(in_features=32*36,out_features=128)
         self.fc2 = nn.Linear(in_features=128, out_features=2)

    #defining the forward pass -> specified how input tensor(x) is transformed through
    #each layer of network
    def forward(self,x):
         x = self.conv1(x) #x = first convulutional layer
         x = nn.ReLU()(x) #applies the ReLU activation function - > non-linearity to output
         x = self.conv2(x)
         x = nn.ReLU()(x)

        #flattens the tensor into 2d tensor, where each row -> sample
        #each column -> feature
         x = x.view(x.size(0),-1)
         x = self.fc1(x) #applies first fully connected layer to flattened tensor
         x = torch.relu(x)
         x = self.fc2(x) #applies final fully connected layers, producing output values
         return x

#create instance of model
model = CNN()

#defining loss function and optimizer(adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #model.parameters() passes all model params to the optimizer. learning rate: 0.001

num_epochs = 10
#training loop
for epoch in range(num_epochs):
     for i, data in enumerate(train_loader):
          inputs, labels = data
          optimizer.zero_grad() #clears gradients of model parameters to prevent accumlation from previous iterations
          
          #forward pass
          outputs = model(inputs)

          #compute the loss
          loss = criterion(outputs,labels)

          #Backpropagation
          loss.backward()

          #update weights                 
          optimizer.step()
