# 함수

# 함수(Funtion) : 비슷한 목적으로 묶어진 약속
# 여러개의 문장을 하나로 묶어서 다시 호출할 수 있도록 이름을 부여
# 함수도 객체처럼 메모리상에 생성
# 직접 만들거나 내장된 방식 불러오는 방법.

## def 로 시작해서 콜론(:) 으로 끝냄
## 함수의 시작과 끝은 들여쓰기로 구분
## 함수 선언을 '헤더 파일'에 미리 선언하거나 인터페이스와 구현으로 나누지 않고
## 필요할 때 바로 선언하고 사용할 수 있음.

def Times(a,b):
    return a*b

print(Times)
print(Times(10,10))

# 내장 함수인 globals()를 통해 메모리에 있는 객체들을 볼 수 있음.
print(globals())

# 함수를 호출한 곳으로 되돌아 가려면?
# 함수에서 return은 함수를 종료하고 해당 함수를 호출한 곳으로 되돌아가게 함
# def setValue(newValue):
#     x = newValue        리턴이 없음.

#retval = setValue(10)
#print(retval)
def swap(x,y):
    return y,x

print(swap(1,2))
a,b = swap(1,2)
print(a)
print(b)