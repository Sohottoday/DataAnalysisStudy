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