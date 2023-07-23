import torch
import torch.optim as optim
import torch.nn as nn
from torch import linalg as LA
from torch import Generator
from time import time

from numbercode import NumberCode 
from dataset import NumberCodeDataSet
from model import NumberCodeModel 
from torch.utils.data import DataLoader
from configparser import ConfigParser

if torch.cuda.is_available():
  print("CUDA is available")
  enviroment = 'cuda'
else:
  enviroment = 'nocuda'
  print("CUDA isn't available")

config = ConfigParser()
config.read('train.conf')

deviceName = config.get(enviroment,'device')   
device = torch.device(deviceName)

batch_size = config.getint(enviroment, 'batch_size')
trainset = NumberCodeDataSet(config.getint(enviroment, 'dataset_size'))
loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
model = NumberCodeModel(hiddenLayerOrder=config.getint(enviroment, 'hidden_layer_size'))
model.to(device)
model.init_origin()

criterion = nn.CrossEntropyLoss()
##optimizer = optim.SGD(model.parameters(), lr=10.0)
lr = config.getfloat(enviroment, 'learning_rate')
optimizer = optim.Adam(model.parameters(), lr)

numberOfEpochs = config.getint(enviroment, 'epochs')
min_validation_loss = 10.0
epochCounter = 0
trainingTime = 0.0
print('Training '+str(numberOfEpochs)+" epochs on "+deviceName+" with learning rate "+str(lr))
for epoch in range(numberOfEpochs):  # loop over the dataset multiple times

    ##Training loop 
    start_time = time()
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        input = data['input']
        input = input.to(device)
        result = data['result']
        result = result.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, result)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batchGroupSize = config.getint(enviroment, 'batch_group_size')
        if i % batchGroupSize == (batchGroupSize -1):    # print every batchGroupSize mini-batches
            print("#############################################################################")
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchGroupSize:.3f}')
            print("Weight norm = "+str(model.params_norm()))
            print("Grads norm = "+str(model.grads_norm()))
            print("#############################################################################")
            running_loss = 0.0
    end_time = time()
    epochCounter+=1
    trainingTime+=(end_time-start_time)
    totalTrainingTime = int(round(trainingTime))
    trainingTimePerEpoch = int(round(trainingTime/float(epochCounter)))
    print("Trained "+str(epochCounter)+" Epochs in "+str(totalTrainingTime)+" seconds, "+str(trainingTimePerEpoch)+" seconds per epoch")

    ##Validation loop
    validation_loss = 0.0
    counter = 0
    validation_set = NumberCodeDataSet(config.getint(enviroment, 'dataset_size'))
    validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = True)
    for data in validation_loader:
        input = data['input']
        input = input.to(device)
        result = data['result']
        result = result.to(device)

        # forward 
        output = model(input)
        loss = criterion(output, result)
        
        validation_loss += loss.item()
        counter +=1
    
    validation_loss = validation_loss/float(counter)
    if (min_validation_loss > validation_loss):
        print("VALIDATION LOSS = "+str(validation_loss))
        torch.save(model.state_dict(), 'saved_model.pth')
        min_validation_loss = validation_loss
        

print('Finished Training with min_validation_loss = '+str(min_validation_loss))




