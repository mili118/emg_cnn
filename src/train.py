import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN
from preprocess import load_data

def train():
     #load data set uc irvine 
     train_loader, val_loader = load_data(batch_size=16)

     #define model and loss optimizer(adam)
     model = CNN()
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

         print(f"Epoch [{epoch+1}/{num_epochs}] completed")

         #save model checkpoints
         torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

         if __name__ == "__main_":
             train()

