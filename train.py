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

if torch.cuda.is_available():
  print("CUDA is available, running on it")
  device = torch.device("cuda")
else:
  print("CUDA isn't available, running on CPU")
  device = torch.device("cpu")

torch.set_default_device(device)

batch_size = 16
trainset = NumberCodeDataSet(100000)
loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, generator = Generator(device))
model = NumberCodeModel(hiddenLayerOrder=4)
model.init_origin()

criterion = nn.CrossEntropyLoss()
##optimizer = optim.SGD(model.parameters(), lr=10.0)
optimizer = optim.Adam(model.parameters(), 0.001)


numberOfEpochs = 1000
min_validation_loss = 10.0
epochCounter = 0
trainingTime = 0.0
for epoch in range(numberOfEpochs):  # loop over the dataset multiple times

    ##Training loop 
    start_time = time()
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        input = data['input']
        result = data['result']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, result)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batchGroupSize = 2000
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
    validation_set = NumberCodeDataSet(10000)
    validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = True)
    for data in validation_loader:
        input = data['input']
        result = data['result']

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




