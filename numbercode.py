from random import randint

class NumberCode:
    def __init__(self, code) -> None:
        self.code = code.strip()
    
    def validate(self):
        if (len(self.code) != 10):
            return False
        return (self.calculateValidationSum() == int(self.code[9]))
        
    def calculateValidationSum(self):
        numbers = []
        for i in range(9):
            numbers.append(int(self.code[i]))
        return max(numbers)
    
    
    def createRandomCode(wrong = False):
        result = ""
        numbers = []
        for i in range(9):
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
