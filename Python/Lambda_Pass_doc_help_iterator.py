# lambda : 익명 함수를 만들 때
# pass : 아무것도 하지 않고 그냥 넘어가도록 할 때
# __doc__, help : 생성한 함수의 설명을 추가하고 싶을 때
# iterator 객체 : 순회 가능한 형식에서 값을 안전하고 빠르게 나열하고 싶을 때

# 람다(lambda) 함수 : 함수의 이름이 없는 바디만 정의한 익명함수를 만들 수 있음
# 일반 함수와 마찬가지로 여러 개의 인자를 전달 받을 수 있음
# return 구문을 적지 않아도 반환 값을 돌려줄 수 있음.

# lambda 인자 : <구문>
g = lambda x,y : x*y        # g라는 변수에 저장해서 계속 호출할 수 있음
print(g(2,3))

print((lambda x : x * x)(3))    # 이 경우는 람다함수를 정의하고 바로 호출하고 사라지기 때문에 계속 호출할 수 없음.

print(globals())

# pass 키워드 : 아무것도 하지 않고 그냥 넘어가도록 할 경우 pass 키워드를 사용해서
# 함수, 모듈, 클래스의 내부를 채울 수 있음.
#while True:
#    pass        # 빠져나가려면 키보드에서 'ctrl + c'를 누름
#KeyboardInterrupt
#class temp:
#    pass

# __doc__ 속성과 help
help(print)

def add(a,b):
    return a+b

print(help(add))

add.__doc__ = "이 함수는 2개의 인자를 받아서 덧셈을 수행하는 함수입니다."
print(help(add))


# 이터레이더(iterator) 객체 : 순회 가능한 객체의 요소를 순차적으로 열거하도록 하는 객체
# 리스트, 튜플, 문자열처럼 순회 가능한 시퀀스 형식에는 구현되어 있음
for element in [1,2,3]:
    print(element)

for element in (1,2,3):
    print(element)

for key in {'one':1, 'two':2}:      # 반복 구문인 for in 루프를 사용해서 각각의 아이템을 순차적으로 접근해서 리턴 할 수 있음.
    print(key)

for char in "123":
    print(char)

s = 'abc'
it = iter(s)        # 내장 함수인 iter()함수를 통해 이터레이터 객체를 만들기
print(it)
print(next(it))     # enxt() 함수를 통해 호출하면 첫번째를 리턴하고 다음번 방에 잇는 값을 리턴하며, 값이 없으면 순회 작업이 중단.

