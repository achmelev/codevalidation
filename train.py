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

NumberCode.codeWith = config.getint(enviroment, 'width')

deviceName = config.get(enviroment,'device')   
device = torch.device(deviceName)

print("Creating data and model...")
batch_size = config.getint(enviroment, 'batch_size')
trainset = NumberCodeDataSet(config.getint(enviroment, 'dataset_size'))
loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
validation_set = NumberCodeDataSet(config.getint(enviroment, 'dataset_size'))
validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = True)
model = NumberCodeModel(hiddenLayerOrder=config.getint(enviroment, 'hidden_layer_size'))
model.to(device)
model.init_origin()
print("Done")

criterion = nn.CrossEntropyLoss()
##optimizer = optim.SGD(model.parameters(), lr=10.0)
lr = config.getfloat(enviroment, 'learning_rate')
optimizer = optim.Adam(model.parameters(), lr)

numberOfEpochs = config.getint(enviroment, 'max_epochs')
stop_limit = config.getint(enviroment, 'early_stopping_epochs')
min_validation_loss = 10.0
epochCounter = 0
stopCounter = 0
trainingTime = 0.0
calculationTime = 0.0
print('Training for max '+str(numberOfEpochs)+" epochs on "+deviceName+" with learning rate "+str(lr))
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

        calc_start_time = time()
        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, result)
        loss.backward()
        optimizer.step()
        calc_end_time = time()
        calculationTime +=(calc_end_time-calc_start_time)

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
    totalCalculationTime = int(round(calculationTime))
    trainingTimePerEpoch = int(round(trainingTime/float(epochCounter)))
    print("Trained "+str(epochCounter)+" Epochs in "+str(totalTrainingTime)+" seconds including "+str(totalCalculationTime)+" seconds calc time, "+str(trainingTimePerEpoch)+" seconds per epoch")

    ##Validation loop
    validation_loss = 0.0
    stop_validation_loss = config.getfloat(enviroment, "stop_validation_loss")
    counter = 0
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
        stopCounter = 0
        print("VALIDATION LOSS = "+str(validation_loss))
        torch.save(model.state_dict(), 'saved_model.pth')
        min_validation_loss = validation_loss
        if (min_validation_loss < stop_validation_loss):
           break
    else:
       print("VALIDATION LOSS = "+str(validation_loss)+" ---> NO IMPROVEMENT")
       stopCounter = stopCounter+1
       if (stopCounter > stop_limit):
          print("Reached stop limit -- STOPPING")
          break
        

print('Finished Training with min_validation_loss = '+str(min_validation_loss))




