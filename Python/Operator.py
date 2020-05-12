# 연산자(Operator)

# 산술 연산자
# ** : 제곱
print("2의 3승 : ",2**3)
print("2의 4승 : ",2**4)

# % : 나머지 값
print("나머지만 구함",5%2)

# // : 몫만 구함
print("몫만 구함", 5//2)

# + : 정수+정수는 정수 연산, 정수+실수는 실수 연산을 함.

# 관계 연산자
# == : 같다, != : 같지 않다, <,<=,>,>= : 대소 비교

# 논리 연산자
x = 10
y = 20
print(x>5 and y<15)
print(x>5 or y<15)
print(not(x>5))
print(not(y<5))

# bool() 함수는 True와 False만 저장하는 자료형
# 각종 내장 형식을 논리연산자에 사용하는 경우
# 숫자형 : True==0 or 0.0이 아닐때, False==0 or 0.0 일 때
# 문자열 : 채워져 있으면 True, 값이 비어 있으면 False
# 리스트, 튜플, 세트, 딕셔너리 : 값이 있으면 True, 값이 없으면 False
print(bool(0),"|", bool(-1), "|", bool('test'),"|", bool(None))
