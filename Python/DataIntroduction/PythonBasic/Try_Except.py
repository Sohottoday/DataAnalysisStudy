# try except 구문을 통한 에러 처리
# 실행시간에 에러가 발생하여 코드가 중단되는 것을 방지할 때 사용(가드레일 역할)
# try:
#   <예외 발생 가능성이 있는 문장>
# except <예외 종류>:
#   <예외 처리 문장>
# except (예외 1, 예외2):       여러개의 에러를 튜플 형태로 나열 가능
#   <예외 처리 문장>
# except 예외 as 인자:
#   <예외 처리 문장>
# else:
#   <예외가 발생하지 않은 경우 수행 될 문장>
# finally:
#   <예외 발생 유무에 상관없이 try블록 이후 수행할 문장>

def divide(a, b):
    return a/b

try:
    c = divide(5, 'string')
except ZeroDivisionError:
    print('두 번째 인자는 0이여서는 안됨')
except TypeError:
    print('모든 인자는 숫자여야 함')
except:
    print('음 무슨 에러인지 모르겠음')
else:
    print('Result:{0}'.format(c))
finally:
    print('항상 finally 블럭은 수행됨')