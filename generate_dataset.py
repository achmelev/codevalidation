from numbercode import NumberCode 
import sys
from random import choice

NumberCode.codeWith = int(sys.argv[1])
number_of_items = int(sys.argv[2])
filename = sys.argv[3]

print("Generating "+str(number_of_items)+" random codes of width "+str(NumberCode.codeWith)+" and writing to "+filename)

f = open(filename,'w')

for i in range(number_of_items):
    wrongValue = choice((True, False))
    code = NumberCode.createRandomCode(wrongValue)
    if ((i+1)%1000 == 0):
        print("Written "+str(i+1)+" codes...")
    f.write(code.code)

f.close()

print('Done!')



