# 파이썬 대표적인 에러들

# 1. 인덱스 에러 : 인덱스 첨자 범위를 벗어난 경우
#a = [10,20,30]
#print(a[3])
#Traceback (most recent call last):
#  File "c:/Users/user/Desktop/Programing/Github/SelfStudy/Python/FunctionError.py", line 5, in <module>
#    print(a[3])
#IndexError: list index out of range

# 2. 형식이 잘못된 경우
#result = 5/'string'
#Traceback (most recent call last):
#  File "c:/Users/user/Desktop/Programing/Github/SelfStudy/Python/FunctionError.py", line 12, in <module>
#    result = 5/'string'
#TypeError: unsupported operand type(s) for /: 'int' and 'str'

# 에러 처리를 위한 클래스 계층
# BaseException
#    SystemExit
#    Exception
#    StopIteration
#    ArithmeticError
