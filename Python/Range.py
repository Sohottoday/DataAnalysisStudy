# 수열 Raange 함수
# 파이썬 내장 함수
# 제어문과 연관된 유용한 함수 :
# range() : 수열의 생성
print(list(range(10)))
print(list(range(5,10)))        # 시작값, 종료값이 있는 경우
print(list(range(10,0,-1)))     # 시작 값, 종료 값, 증가 값이 있음
print(list(range(10,20,2)))     # 10에서 20까지 짝수만 출력

# 리스트 내장 방식 : 기존의 리스트 객체를 이용해 조합, 필터링 등의 추가적인 연산을 통해
# 새로운 리스트 객체를 생성하는 경우 '리스트 내장'이 매우 효율적임.
# <표현식> for <아이템> in <시퀀스 타입 객체> (if <조건식>)
I = [1,2,3,4,5]
print([i ** 2 for i in I])

t = ("apple", "banana", "orange")
print([len(i) for i in t])

d = {100:"apple", 200:"banana", 300:"orange"}
print([v.upper() for v in d.values()])

print( [i**3 for i in range(5)] )

# 반복문 작성에 도움이 되는 함수 filter
# filter()함수는 조건에 해당하는 함수의 이름을 넘겨주면 해당 함수를 통해
# 걸러내기(필터링)을 해주는 내장 함수
# filter(<function> | None, <이터레이션이 가능한 자료형>)
L = [10,25,30]
IterL = filter(None, L)
for i in IterL:
    print("Item:{0}".format(i))

def GetBiggerThan20(i):     # filter()함수에 첫번째 인자로 GetBiggerThan20()함수를 넘겨주면
    return i > 20           # 내부의 논리식에서 참을 리턴하는 경우만 for in 루프에서 포함되어 출력
IterL = filter(GetBiggerThan20, L)
for i in IterL:
    print("Item:{0}".format(i))
    