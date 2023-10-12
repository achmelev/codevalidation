from numbercode import NumberCode, createRandomCode
from random import randint
import unittest



class TokenTreeTest(unittest.TestCase):

    def setUp(self) -> None:
        NumberCode.codeWith = 5
    
    def test_numbercode(self):
        for i in range(10):
            rint = randint(0,10)
            rWrong = rint > 5
            code = createRandomCode(wrong = rWrong)
            self.assertEqual(code.validate(), (not rWrong))

if __name__ == '__main__':
    unittest.main()