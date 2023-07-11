import torch
from torch.utils.data import Dataset, DataLoader
from numbercode import NumberCode
from random import choice

class NumberCodeDataSet(Dataset):

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

    def __init__(self, len):
        self.len = len
        self.samples = []
        for i in range(self.len):
            wrongValue = choice((True, False))
            code = NumberCode.createRandomCode(wrongValue)
            self.samples.append([NumberCodeDataSet.createInputAsFloatList(code.code), NumberCodeDataSet.createOutputAsIndex(not(wrongValue))])
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'input': torch.FloatTensor(self.samples[idx][0]), 'result': self.samples[idx][1]}
        return sample