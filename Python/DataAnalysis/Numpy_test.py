import numpy as np

list1 = [1, 2, 3, 4]
a = np.array(list1)
print(a)

print(a.shape)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

print(b.shape)

aa = np.zeros((2, 2))	# 2행 3열의 매트릭스를 0으로 다 채운다.
print(aa)
print(type(aa))

aa = np.ones((2, 3))	# 2행 3열의 매트릭스를 1로 다 채운다.
print(aa)

aa = np.full((2, 3), 10)	# 2행 3열의 매트릭스를 10으로 다 채운다.
print(aa)

aa = np.eye(4)		# 4행 4열을 의미하며 대각선에 값을 넣는다.
print(aa)

aa = np.array(range(20)).reshape((5, 4))
print(aa)			# 0부터 19까지 생성한 뒤 5행 4열의 매트릭스에 넣는다.

aa = np.array(range(15)).reshape((3, 5))
print(aa)		# 0부터 15까지 생성한 뒤 3행 5열의 매트릭스에 넣는다.

list2 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

arr = np.array(list2)

a = arr[0:2, 0:2]

print(a)

b = arr[1:, 1:]
print(b)

list3 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

a = np.array(list3)

# 정수 인덱싱
res = a[[0, 2], [1, 3]]     # 즉 배열 기준 0행 1열 값과 2행 3열 값을 가져오라는 의미.
print(res)

# boolean 인덱싱

list4 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

aa = np.array(list4)

b_arr = np.array([
    [False, True, False],
    [True, False, True],
    [False, True, False]
])      # False는 선택하지 않고 True는 선택하겠다는 의미.

n = aa[b_arr]
print(n)

# 표현식을 통한 boolean indexing 배열 생성
## 배열 aa에 대해서 짝수인 배열 요소만 True로 지정하겠다는 가정
b_arr = (aa % 2==0)
print(b_arr)

print(aa[b_arr])

aaa = aa[aa%2 == 0]
print(aaa)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b
print(c)

c = np.add(a, b)
print(c)

# 리스트와는 계산 결과 값부터 다르다.
d = [1, 2, 3]
e = [4, 5, 6]
f = d + e
print(f)

c = a - b
print(c)

c = np.subtract(a, b)
print(c)

#c = a * b
c = np.multiply(a, b)
print(c)

#c = a/b
c = np.divide(a, b)
print(c)
print('-----')
# 2차원 배열의 곱 / 매트릭스 배열의 곱
# 1차원 배열을 벡터라 하고 2차원 이상의 배열을 매트릭스라 한다.
list11 = [
    [1, 2],
    [3, 4]
]

list12 = [
    [5, 6],
    [7, 8]
]

a = np.array(list11)
b = np.array(list12)

# numpy에서 vector와 matrix의 product를 구하기 위해서 dot() 함수를 이용한다.
product = np.dot(a, b)
print(product)

s = np.sum(a)
print(s)

s = np.sum(a, axis = 0)
print(s)
s = np.sum(a, axis = 1)
print(s)

p = np.prod(a)
print(p)

p = np.prod(a, axis = 0)
print(p)

p = np.prod(a, axis = 1)
print(p)


x = np.float32(1.0)
print(x)
print(type(x))
print(x.dtype)

z = np.arange(5, dtype='f')    # range 함수와 비슷하지만 나란히 정렬하여 배열을 만든다. 데이터 타입 설정이 가능하다.
print(z)
bb = np.arange(3, 10)
print(bb)
cc = np.arange(3, 10, dtype=np.float)
print(cc)
dd = np.arange(2, 3, 0.1)
print(dd)
print(dd.dtype)

aa = np.array([1, 2, 3], dtype='f')
print(aa.dtype)

xx = np.int8(aa)
print(xx)
print(xx.dtype)


q = np.array([[1, 2], [3, 4]])
w = 10
y = np.array([10, 20])

z = q * w
print(z)

z = q * y
print(z)

qq = np.array([[11, 21], [34, 43], [0, 9]])
print(qq)
print(qq[0][1])

for row in qq:
    print(row)

qq = qq.flatten()
print(qq)

print(qq[np.array([1, 3, 5])])
print(qq[qq>25])    # numpy에 부등호 연산자를 사용할 경우 True False로 값이 나온다.
print(qq > 25)

print(qq.ndim)  
print(qq.itemsize)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a)

# 인덱스를 저장할 배열 생성
idx = np.array([0, 2, 0, 1])
print(idx)

print(a[np.arange(4), idx])

print(a[np.arange(4), idx] * 2)

# 벡터의 내적
x = np.array([[1, 2], [3,4]])
v = np.array([9, 10])
w = np.array([11, 12])

print(v.dot(w))     # 또는 np.dot(v, w)
# v[0] * w[0] + v[1] * w[1]

# 매트릭스와 벡터의 곱
print(x.dot(v))
#x[0,0] * v[0] + x[0,1] * v[1] , x[1,0] * v[0] + x[1,1] * v[1]

# 전치 행렬의 표현은 T속성을 이용한다.
tt = np.array([[1, 2], [3, 4]])
print(tt)
print(tt.T)

# 배열 생성 초기화, 값들을 모두 초기화시킨다.
g = np.empty((4, 3))
print(g)

# 3차원 배열 만들기
d = np.array([[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
                [[11, 12, 13, 14],
                [15, 16, 17, 18],
                [19, 20, 21, 22]]])

print(d)

# len : 3차원 배열의 행, 열. 깊이 구하는 방법
print(len(d))        # 3차원 배열의 깊이
print(len(d[0]))     # 3차원 배열의 행
print(len(d[0][0]))  # 3차원 배열의 열

c = np.ones((2, 3, 4), dtype="i")
print(c)

# ones_like() : 지정한 배열과 똑같은 크기의 배열을 만든다
# (copy와 다른점은 dtype의 설정이 가능하므로 같은 크기의 다른 종류의 배열을 만들 수 있다.)
k = np.ones_like(c)

cc = np.copy(c)


# 차원 축소 연산(dimension reduction 연산)
## 행렬의 하나의 행에 있는 원소들을 하나의 데이터 집합으로 보고 그 집합의 평균을 구하면
##각 행에 대해 하나의 숫자가 나오게 되는데 이 경우 차원 축소가 된다. 이러한 연산을 차원 축소 연산이라 한다.

# Numpy에서의 차원 축소 연산 명령 또는 메서드
# 최대/최소 : min, max, argmin, argmax
# 통계 : sum, mean, median, std, var
# boolean : all, any

x = np.array([1,10, 100])
print(x)
print(x.min())

# argmin : 최소값의 위치
# argmax : 최대값의 위치
print(x.argmin())
print(x.argmax())

# mean : 평균값
# median : 중간값, 여러개의 숫자들이 있을 때 크기순으로 정렬한 다음 가장 가운데 위치한 숫자
# 개수가 짝수개일 경우 중간의 좌우값의 평균값을 반환한다.
# std : 
# var : 
print(x.mean())
print(np.median(x))

# all : 모든 조건이 들어 맞을때
# any : 하나의 조건이라도 들어 맞을때

print(np.all([True, True, False]))
print(np.any([False, False, True, False]))

# axis 속성을 부여하는 메서드들은 대부분 차원 축소 명령에 속한다.

# 정렬
# sort명령이나 메서드를 사용하여 배열안의 원소를 크기에 따라 정렬하여 새로운 배열을 만들 수 있다.
# 2차원 이상인 경우에는 행이나 열을 각각 따로 정렬할 수 있는데 이때, axis 속성을 사용하여 행과 열을 결정할 수 있다.

a = np.array([[4, 3, 5, 7],
            [1, 12, 11, 9],
            [2, 15, 1, 14]
            ])
print(np.sort(a))   # default값이 1이다
print(np.sort(a, axis=0))
print(a.argsort())  # 자료를 정렬하는 것이 아닌 순서만 알고싶을 때 사용된다.

# 배열 더하기(합성)
a = np.array([1, 2, 3])
b = np.array([3, 2, 3])
# column_stack() 배열을 열 기준으로 합침, 3개 이상도 사용 가능하다.
print(np.column_stack((a, b)))

# 배열 나누기(쪼개기)
# split = array_split

x = np.arange(9.0)
print(x)

x_1 = np.split(x, 3)
# x_1 = np.array_split(x, 3)
print(x_1)

print(x_1[1])

x2 = np.split(x, [3, 4])    # 인덱스 3위치에서 한번 나누고 4위치에서 한번 나누겠다는 의미
# 이런식으로 자신이 원하는대로 자를 수 있다.
print(x2)

# dsplit : 3차원 배열 나누기
y = np.arange(16).reshape(2, 2, 4)
print(y)

print(np.dsplit(y, 2))
print('------------')

# hsplit : 3차원 배열 열 기준으로 나누기
print(np.hsplit(y, 2))
print("----------")

# vsplit: 수직 기준으로 3차원 배열 나누기
print(np.vsplit(y, 2))

# 반복 생성
a = np.array([0, 1, 2])
print(np.tile(a, 2))
print(np.tile(a, (2, 2)))

b = np.array([[1, 2], [3, 4]])
print(np.tile(b, 2))
print(np.tile(b, (2,2)))

# 배열 반복
print(np.repeat(3, 4))  # 3을 4번 반복하는 배열 생성
x = np.array([[1, 2], [3, 4]])
print(np.repeat(x, 2))

print(np.repeat(x, 3, axis=0))