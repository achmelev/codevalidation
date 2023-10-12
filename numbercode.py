from random import randint

def calculateValidationSum(code, index):
    sum = 0
    for i in range(index*9, (index+1)*9):
        sum+=int(code[i])
    return sum%10

def createRandomCode(wrong = False):
    result = ""
    numbers = []
    for i in range(NumberCode.codeWith*9):
        digit = randint(0,9)
        numbers.append(digit)
        result+=str(digit)
    if (wrong):
        while (True):
            candidate = result
            for index in range(NumberCode.codeWith):
                candidate+=str(randint(0,9))
            code = NumberCode(candidate)
            if (not code.validate()):
                return code
    else:
        for index in range(NumberCode.codeWith):
            candidate+=str(calculateValidationSum(result, index))
        return NumberCode(result)

class NumberCode:

    codeWith = 5

    def __init__(self, code) -> None:
        self.code = code.strip()
        if (len(self.code) != self.codeWith*10):
            raise Exception("Wrong code with "+str(len(self.code))+'instead of '+str(self.codeWith*10)) 
    
    def validate(self):
        result = True
        for index in range(self.codeWith):
            result = result and calculateValidationSum(self.code, index) == int(self.code[9*self.codeWith+index])
        return result
        
    
    
