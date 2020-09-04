
su = 2 + 3 * 5
print(su)

su = (2 + 3) * 5
print(su)

# = 는 대입 연산자라고 한다.(우선순위가 마지막)

# 비교(관계) 연삱 : > >= < <= == !=
# 연산의 결과물은 반드시 True 또는 False가 된다.
# 제어문(if, for 구문 등등)에서 많이 사용되므로 중요!
a = 10
b = 20
result = a >= b
print(result)

result = a < b
print(result)

result = a == b
print(result)

result = a != b
print(result)

# 논리 연산자 : and, or, not
a = 10
b = 20
first = a>=b        # False
second = a != b     # True

result = first and second
print(result)

# 연산자 우선 순우 : (), * 또는 /, + 또는 -, 관계 연산, not, and, or, ... 대입

third = a == b
result = first and second or third
print(result)

# 
result = not(result)
print(result)
