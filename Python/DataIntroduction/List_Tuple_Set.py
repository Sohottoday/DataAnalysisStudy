# 2020.05.11
# List, Tuple, Set

# List 관련 메소드
colors = ['red', 'grren', 'gold']
print(colors)

# append() : 기존 리스트에 값을 추가할 때
colors.append('blue')
print(colors)

# insert() : 원하는 위치에 추가할 때
colors.insert(1,'black')
print(colors)

# index() : 특정 값의 위치를 확인할 때
print(colors.index('red'))
# 리스트 내 같은 값이 여러개일 때 colors.index('red',1) 과 같은 방식으로 확인 가능

# count() : 현재 해당 값의 개수를 반환할 때
print(colors.count('red'))

# pop() : 해당 값의 값을 뽑을 때
colors.pop(1)
print(colors)
# colors.pop() 과 같이 값을 주지 않으면 가장 뒤의 값을 뽑아냄(즉 추출,삭제)

# remove() : 특정 값을 삭제할 때
colors.remove('gold')
print(colors)

# sort() : 정렬할 때
colors.sort()
# reverse() : 정렬할 때 (반대반향)
colors.reverse()


# 세트 : 수학시간에 배운 집합과 유사한 형태 / 별도의 순서는 없음
a = {1,2,3}
b = {3,4,5}
print(type(a))
print(a)

# 세트는 중복값을 허용하지 않는다
# ex) a={1,2,3,3,4,4,5}  => a={1,2,3,4,5}

# 합집합
print(a.union(b))

# 교집합
print(a.intersection(b))
print(a & b)

# 차집합
print(a-b)
print(a.difference(b))

# 합집합
print(a|b)


# 튜플(tuple) : 리스트와 유사, 리스트와 달리 []대신 ()로 묶어 표현, 읽기 전용
# 제공되는 메소드는 리스트에 비해 적지만 속도는 보다 빠름
t = (1,2,3)
print(type(t))
d,f = 1,2
print(d,f)
(d,f) = (1,2)
print(d,f)

# 튜플이 자주 사용되는 경우
# 함수에서 하나 이상의 값을 리턴하는 경우
def calc(d,f):
    return d+f, d*f
# def 문 : 함수 정의하기(파이썬에서는 함수를 정의할 때 def를 사용)
j,k = calc(5,4)

# 함수에서 하나 이상의 값을 출력하는 경우
print('id: %s, name: %s'% ('jun','전우치'))

# 튜플에 있는 값을 함수 인수로 사용하는 경우
yyy=(4,5)
print(calc(*yyy))


# 리스트, 세트, 튜플은 생성자 list(), set(), tuple()을 이용해 언제든지 서로 변환될 수 있다.
q = set((1,2,3))
print(type(q))
w = list(q)
print(type(w))
e = tuple(w)
print(type(e))
# 파이썬 쉘에서 진행하면 print를 안해도 type()같은건 출력 가능한지?



