# 사용자 정의 예외 클래스

# raise 구문 : 사용자가 강제로 예외를 일으킬 때 사용하는 구문
def raiseErrorFunc():
    raise NameError

try:
    raiseErrorFunc()
except:
    print("NameError is Catched")

# 사용자 정의 예외 클래스 사용
# 내장 예외만으로는 한계가 있기 때문에 사용자 정의 예외를 정의할 수 있음.
class NegativeDivisionError(Exception):
    def __init__(self, value):
        self.value = value

def positiveDivide(a,b):
    if(b < 0):      #0보다 적은 경우 NegativeDivisionError 강제로 발생시킴
        raise NegativeDivisionError(b)
    return a/b
# try 블록 안에서 함수를 호출하고 사용자 정의 에러 클래스로 예외를 처리하는 데모
try:
    ret = positiveDivide(10, -3)
    print('10/3 = {0}'.format(ret))
except NegativeDivisionError as e:
    print('Error - Second argument of PositiveDivide is', e.value)
except ZeroDivisionError as e:
    print('Eroor - ',e.args[0])
