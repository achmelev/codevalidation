from random import randint

class NumberCode:

    codeWith = 10

    def __init__(self, code) -> None:
        self.code = code.strip()
        if (len(self.code) != self.codeWith):
            raise Exception("Wrong code with "+len(self.code)+'instead of '+self.code.width) 
    
    def validate(self):
        return (self.calculateValidationSum() == int(self.code[self.codeWith-1]))
        
    def calculateValidationSum(self):
        numbers = []
        for i in range(self.codeWith-1):
            numbers.append(int(self.code[i]))
        return max(numbers)
    
    
    def createRandomCode(wrong = False):
        result = ""
        numbers = []
        for i in range(NumberCode.codeWith-1):
            digit = randint(0,9)
            numbers.append(digit)
            result+=str(digit)
        if (wrong):
             right = max(numbers)
             candidate = randint(0,8)
             if (candidate >= right):
                 candidate = candidate+1
             result+=str(candidate)
        else:
            result+=str(max(numbers))
        return NumberCode(result)
