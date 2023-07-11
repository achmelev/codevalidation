import torch
from random import choice
from model import NumberCodeModel
from numbercode import NumberCode 
from dataset import NumberCodeDataSet
import torch.nn as nn

model = NumberCodeModel(hiddenLayerOrder=4)

model.load_state_dict(torch.load("saved_model.pth"))

counter = 0
counterSuccess = 0

for i in range(1000):
    wrongValue = choice((True, False))
    code = NumberCode.createRandomCode(wrong=wrongValue)
    counter = counter+1
    input = torch.FloatTensor(NumberCodeDataSet.createInputAsFloatList(code.code))
    output = model.forward(input)
    softmax = nn.Softmax(dim = 0)
    expectedResult = not(wrongValue)
    tensorResult = softmax(output)
    calculatedResult = tensorResult.data[1] > tensorResult.data[0]
    if (calculatedResult == expectedResult):
        counterSuccess = counterSuccess+1

print(str(counterSuccess)+" successes from "+str(counter)+" trials")
    
    
