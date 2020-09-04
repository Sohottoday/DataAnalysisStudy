
# 튜플(tuple) : 목록이라는 측면에서 리스트와 유사
# 단, 값에 대하여 읽기 전요이고, 리스트보다는 빠르다.
# 관련 함수 : tuple()
# 인덱싱, 슬라이싱 가능하다.
# 소괄호를 사용한다.

tuple1 = (1, 2, 3)

tuple1 = tuple1 + (4, )    # 튜플 합치기
print(tuple1)

tuple2 = 1, 2, 3, 4
mylist = [1, 2, 3, 4]
tuple3 = tuple(mylist)

if tuple2 == tuple3:
    print('yes')
else:
    print('no')

tuple4 = (1, 2, 3)
tuple5 = (4, 5, 6)

# + 연산자 : 튜플을 합쳐주는 기능
# * 연산자 : 튜플을 반복해주는 기능
tuple6 = tuple4 + tuple5
print(tuple6)

tuple7 = tuple4 * 3
print(tuple7)

# swap : 두 변수 간의 값을 상호 맞바꿈
a, b = 10, 20
a, b = b, a
print(a)
print(b)

# 슬라이싱
tuple8 = (1, 2, 3, 4, 5, 6)
print(tuple8[1:3])
print(tuple8[3:])
# tuple8[0] = 100


