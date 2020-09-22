# 매개변수가 2개이고, 둘 다 필수사항이다.

def func01(a, b):
    return 2*a+b

print(func01(3, 5))

# TypeError: func01() takes 2 positional arguments but 3 were given
# 매개 변수 갯수 초과
# print(func01(3, 5, 3))

# TypeError: func01() missing 2 required positional arguments: 'a' and 'b'
# 필수 매개 변수 2개가 누락되었다.
# print(func01())

# 옵션 매개 변수(필수사항이 아니다.)
# 입력되지 않으면 기본값을 사용하겠다는 의미
def func02(a=2, b=1):
    return 2*a+b

print(func02())
print('-'*20)

print(func02(3))
print('-'*20)

print(func02(3, 5))
print('-'*20)

# keyword arguments(키워드 기반 매개변수)
print(func02(b=2, a=1))
print('-'*20)

print(func02(2, b=1))
print('-'*20)

#print(func02(b=1, 2))      #SyntaxError: positional argument follows keyword argument
#print('-'*20)
