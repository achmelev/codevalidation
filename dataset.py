import torch
from torch.utils.data import Dataset, DataLoader
from numbercode import NumberCode
from random import choice
from os.path import getsize


def createInputAsFloatList(code):
    result = []
    for c in code:
        result.append(float(c)/10.0)
    return result

def createOutputAsIndex(result):
    if (result):
        return 1
    else:
        return 0


class NumberCodeDataSet(Dataset):

    def __init__(self, len):
        self.len = len
        self.samples = []
        for i in range(self.len):
            wrongValue = choice((True, False))
            code = NumberCode.createRandomCode(wrongValue)
            self.samples.append([createInputAsFloatList(code.code), createOutputAsIndex(not(wrongValue))])
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'input': torch.FloatTensor(self.samples[idx][0]), 'result': self.samples[idx][1]}
        return sample

class FileNumberCodeDataSet(Dataset):

    def __init__(self, filename):
        size = getsize(filename)
        
        if (size%(NumberCode.codeWith*10+1) != 0):
            raise Exception("File size isn't a multiple of "+str(NumberCode.codeWith*10+1))
        self.len = int(getsize(filename)/(NumberCode.codeWith*10+1))
        self.filename = filename
        self.file = None
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if (self.file == None):
            self.file = open(self.filename,'r')
        self.file.seek((NumberCode.codeWith*10+1)*idx)
        itemStr = self.file.read(NumberCode.codeWith*10+1)
        input = createInputAsFloatList(itemStr[:NumberCode.codeWith*10])
        result = int(itemStr[NumberCode.codeWith*10])
        sample = {'input': torch.FloatTensor(input), 'result': result}
        return sample
