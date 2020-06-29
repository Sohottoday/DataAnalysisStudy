# Numpy

- Numpy는 과학 계산을 위한 라이브러리로 다차원 배열을 처리하는데 필요한 여러 기능을 제공한다.
  - pip3 install numpy
  - 벡터 산술연산
  - 다차원 배열(ndarray)
  - 표준 수학 함수
  - 선형대수, 난수

## numpy배열

- numpy에서 배열은 동일한 타입의 값들을 갖는다. 

  배열의 차원을 rank라고 한다.

  rank를 알아보기 위해서는 ndim 속성을 활용한다.

- shape : 각 차원의 크기를 튜플로 표시한 것

  - ex) 2행, 3열인 2차원 배열의 rank는 2이고, shape(2, 3)

- itemsize : 배열의 각 요소의 바이트에서의 사이즈

  - float64 형식 요소의 배열은 itemsize 8 (64/8)를 가지고 complex32 형식의 요소를 갖는다면 itemsize 4를 갖게 된다.

- numpy 배열 생성

  - 파이썬의 리스트를 사용하는 방법

    array() 함수의 인자로 리스트를 넣어 생성한다.

  ```python
  numpy.array([1, 2, 3])
  ```

  ```python
  import numpy as np
  list1 = [1, 2, 3, 4]
  a = np.array(list1)
  print(a)
  # [1 2 3 4]
  print(a.shape)
  # (4,)
  b = np.array([[1, 2, 3], [4, 5, 6]])
  print(b)
  #[[1 2 3]
  # [4 5 6]]
  print(b.shape)
  #(2, 3)
  b[0, 0]
  # 1
  print(b.ndim)
  ```

  - numpy에서 제공하는 함수를 사용하는 방법

    zeros()함수는 배열에 모두 0을 집어 넣고,

    ones()함수는 모두 1을 집어 넣는다.

    full()함수는 사용자가 지정한 값을 넣는데 사용하고,

    eye()함수는 대각선으로는 1이고 나머지는 0

  ```python
  import numpy as np
  
  aa = np.zeros((2, 2))	# 2행 3열의 매트릭스를 0으로 다 채운다.
  print(aa)
  #[[0. 0.]
  # [0. 0.]]
  print(type(aa))
  # <class 'numpy.ndarray'>
  
  aa = np.ones((2, 3))	# 2행 3열의 매트릭스를 1로 다 채운다.
  print(aa)
  #[[1. 1. 1.]
  # [1. 1. 1.]]
  
  aa = np.full((2, 3), 10)	# 2행 3열의 매트릭스를 10으로 다 채운다.
  print(aa)
  #[[10 10 10]
  # [10 10 10]]
  
  aa = np.eye(4)		# 4행 4열을 의미하며 대각선에 값을 넣는다.
  print(aa)
  #[[1. 0. 0. 0.]
  # [0. 1. 0. 0.]
  # [0. 0. 1. 0.]
  # [0. 0. 0. 1.]]
  
  aa = np.array(range(20)).reshape((5, 4))
  print(aa)			# 0부터 19까지 생성한 뒤 5행 4열의 매트릭스에 넣는다.
  #[[ 0  1  2  3]
  # [ 4  5  6  7]
  # [ 8  9 10 11]
  # [12 13 14 15]
  # [16 17 18 19]]
  
  aa = np.array(range(15)).reshape((3, 5))
  print(aa)		# 0부터 15까지 생성한 뒤 3행 5열의 매트릭스에 넣는다.
  #[[ 0  1  2  3  4]
  # [ 5  6  7  8  9]
  # [10 11 12 13 14]]
  ```



- reshape() : 다차원으로 변형하는 함수



### numpy 슬라이싱, 인덱싱, 연산

- 슬라이싱된 배열은 원본 배열과 같은 데이터를 참조하기 때문에, 슬라이싱된 배열을 수정하면 원본 배열 역시 수정된다.

- numpy배열을 슬라이싱하면, 그 결과는 언제나 원본 배열의 부분배열이다. 그러나, 정수 배열 인덱싱을 하는 경우에는 원본과 다른 배열을 만들 수 있다.

- 배열 인덱싱(array indexing)은 팬시 인덱싱(fancy indexing)이라고도 한다.

- 배열 인덱싱의 유용한 기능 중의 하나는 행렬의 각 행에서 하나의 요소를 선택하거나 바꾸는 기능

- numpy 슬라이싱

  ```python
  import numpy as np
  
  list2 = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
  ]
  
  arr = np.array(list2)
  
  a = arr[0:2, 0:2]
  print(a)
  # [[1 2]
  # [4 5]]
  
  b = arr[1:, 1:]
  print(b)
  #[[5 6]
  # [8 9]]
  ```

  

- numpy 정수 인덱싱(integer indexing)

  - numpy배열 a에 대해서 a[[row1, row2], [col1, col2]]는 a[row1, col1]과 a[row2, col2]

  ```python
  import numpy as np
  
  list3 = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]
  ]
  
  a = np.array(list3)
  
  # 정수 인덱싱
  res = a[[0, 2], [1, 3]]     # 즉 배열 기준 0행 1열 값과 2행 3열 값을 가져오라는 의미.
  print(res)
  #[ 2 12]
  ```

  

- numpy boolean 인덱싱(boolean indexing)

  ```python
  import numpy as np
  
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
  # [2 4 6 8]
  
  # 표현식을 통한 boolean indexing 배열 생성
  ## 배열 aa에 대해서 짝수인 배열 요소만 True로 지정하겠다는 가정
  b_arr = (aa % 2==0)
  print(b_arr)
  #[[False  True False]
  # [ True False  True]
  # [False  True False]]
  
  print(aa[b_arr])
  # [2 4 6 8]
  
  aaa = aa[aa%2 == 0]		# 익숙해지면 이렇게 간단한 표현도 가능하다.
  print(aaa)
  # [2 4 6 8]
  ```




### numpy 연산

- 연산자를 이용할 경우에는 +, -, *, /

  - 배열 a와 배열 b가 있을 때, a + b는 a[0] + b[0], a[1] + b[1] ... 와 같은 방식으로 결과를 리턴

- 함수를 사용할 경우에는 add(), substract(), multiply(), divide()

  ```python
  import numpy as np
  
  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])
  
  c = a + b
  print(c)
  # [5 7 9]
  
  c = np.add(a, b)
  print(c)
  # [5 7 9]
  
  # 리스트와는 계산 결과 값부터 다르다.
  d = [1, 2, 3]
  e = [4, 5, 6]
  f = d + e
  print(f)
  # [1, 2, 3, 4, 5, 6]
  
  c = a - b
  print(c)
  # [-3 -3 -3]
  
  c = np.subtract(a, b)
  print(c)
  # [-3 -3 -3]
  
  #c = a * b
  c = np.multiply(a, b)
  print(c)
  # [ 4 10 18]
  
  #c = a/b
  c = np.divide(a, b)
  print(c)
  # [0.25 0.4  0.5 ]
  ```

  

- 1차원 배열을 벡터라 하고 2차원 이상의 배열을 매트릭스라 한다.

- 2차원 배열의 곱	multiply()가 아니라 **product** 라고 한다

  - a, b	*	x, y		=	ax + bw, ay +bz

    c, d		  w, z	          cx + dw, cy + cz

- numpy에서 vector와 matrix의 product를 구하기 위해서 **dot()** 함수를 이용한다.

  ```python
  # 벡터의 내적
  x = np.array([[1, 2], [3,4]])
  v = np.array([9, 10])
  w = np.array([11, 12])
  
  print(v.dot(w))     # 또는 np.dot(v, w)
  # v[0] * w[0] + v[1] * w[1]
  # 219
  
  # 매트릭스와 벡터의 곱
  print(x.dot(v))
  #x[0,0] * v[0] + x[0,1] * v[1] , x[1,0] * v[0] + x[1,1] * v[1]
  # [29 67]
  
  
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
  #[[19 22]
  # [43 50]]
  ```

  

- numpy에서는 배열간의 연산을 위한 여러 함수들을 제공한다.

  - sum() : 각 배열의 요소를 더하는 함수
  - prod() : 배열의 요소들을 곱하는 함수
  - 이 함수들은 axis 옵션을 사용한다. axis 0이면 컬럼끼리 더하고, 1이면 행끼리 더한다.

  ```python
  list11 = [
      [1, 2],
      [3, 4]
  ]
  a = np.array(list11)
  
  s = np.sum(a)
  print(s)
  # 10
  
  s = np.sum(a, axis = 0)
  print(s)
  # [4 6]
  
  s = np.sum(a, axis = 1)
  print(s)
  # [3 7]
  
  p = np.prod(a)
  print(p)
  # 24
  
  p = np.prod(a, axis = 0)
  print(p)
  # [3 8]
  
  p = np.prod(a, axis = 1)
  print(p)
  # [ 2 12]
  ```




### numpy 자료형(data Type)

- int, float, bool(True/False), complex

  - 정수형(int : integer)
    - int8(-127 ~ -127), int16(-32768 ~ -32767), int32, int64 (부호가 있는 정수형)
    - uint (Unsigned integer : 부호가 없는 정수형) : uint8 (0 ~ 255), unit16(0 ~ 65535), unit32, unit64
  - 실수형 (float)
    - float16, float32, float64
  - 복소수형 (complex)
    - complex64 : 두개의 32비트 부동소수점으로 표시되는 복소수
    - complex128 : 두개의 64비트 부동 소수점으로 표시되는 복소수

- 데이터의 type을 알아보기 위한 dtype

  - 데이터의 타입을 알아볼 수 있다.

  ```python
  import numpy as np
  
  x = np.float32(1.0)
  print(x)
  # 1.0
  print(type(x))
  # <class 'numpy.float32'>
  print(x.dtype)
  # float32
  ```

  - 데이터 타입을 지정해줄 수 있으며 데이터 타입 변환도 가능하다.

  ```python
  aa = np.array([1, 2, 3], dtype='f')
  print(aa.dtype)
  # float32
  
  xx = np.int8(aa)
  print(xx)
  # [1 2 3]
  print(xx.dtype)
  # int8
  ```

  - 데이터 타입을 활용한 **arange()** 함수
    - range 함수와 비슷하지만 나란히 정렬하여 배열을 만든다. 데이터 타입 설정이 가능하다.

  ```python
  z = np.arange(5, dtype='f')    # range 함수와 비슷하지만 나란히 정렬하여 배열을 만든다. 데이터 타입 설정이 가능하다.
  print(z)
  # [0. 1. 2. 3. 4.]
  
  bb = np.arange(3, 10)
  print(bb)
  # [3 4 5 6 7 8 9]
  cc = np.arange(3, 10, dtype=np.float)
  print(cc)
  # [3. 4. 5. 6. 7. 8. 9.]
  
  dd = np.arange(2, 3, 0.1)
  print(dd)
  # [2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9]
  print(dd.dtype)
  # float64
  ```




#### numpy 브로드캐스팅(broadcasting)

- 배열의 차원의 크기가 서로 다른 배열에서도 산술 연산을 가능하게 하는 것

  1 2	*	10	=	10 20

  3 4						 30 40

- 더 높은 쪽의 차원의 형태로 계산된다.

  ```python
  import numpy as np
  
  q = np.array([[1, 2], [3, 4]])
  w = 10
  y = np.array([10, 20])
  
  z = q * w
  print(z)
  #[[10 20]
  # [30 40]]
  
  z = q * y
  print(z)
  #[[10 40]
  # [30 80]]
  
  qq = np.array([[11, 21], [34, 43], [0, 9]])
  print(qq)
  #[[11 21]
  # [34 43]
  # [ 0  9]]
  
  print(qq[0][1])
  # 21
  
  for row in qq:
      print(row)
  #[11 21]
  #[34 43]
  #[0 9]
  ```

  



#### 2차원 배열을 1차원 배열로 변환(평탄화) : flatten()

```python
qq = qq.flatten()
print(qq)
# [11 21 34 43  0  9]

print(qq[np.array([1, 3, 5])])
# [21 43  9]
print(qq[qq>25])    # numpy에 부등호 연산자를 사용할 경우 True False로 값이 나온다.
# [34 43]
print(qq > 25)
# [False False  True  True False False]
```



#### 전치

- 전치 행렬의 표현은 T속성을 이용한다.

``` python
tt = np.array([[1, 2], [3, 4]])
print(tt)
# [[1 2]
#  [3 4]]

print(tt.T)
# [[1 3]
#  [2 4]]
```





- empty() : 배열 생성 초기화, 값들을 모두 초기화시킨다.

``` python
g = np.empty((4, 3))
print(g)
#[[3.56043053e-307 1.60219306e-306 2.44763557e-307]
# [1.69119330e-306 1.78020169e-306 1.33511562e-306]
# [1.11258277e-307 1.33511562e-306 8.01097889e-307]
# [1.33512308e-306 9.79103798e-307 1.24610927e-306]]
```



### 다차원 배열

#### 3차원 배열

- 3차원 배열 만들기

``` python
d = np.array([[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
                [[11, 12, 13, 14],
                [15, 16, 17, 18],
                [19, 20, 21, 22]]])
print(d)
#[[[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# [[11 12 13 14]
#  [15 16 17 18]
#  [19 20 21 22]]]
```



- len() : 3차원 배열의 행, 열, 깊이 구하는 방법

``` python
print(len(d))		# 3차원 배열의 깊이
# 2
print(len(d[0]))	# 3차원 배열의 행
# 3
print(len(d[0][0]))	# 3차원 배열의 열
# 4
```



- ones_like() : 지정한 배열과 똑같은 크기의 배열을 만든다.

  copy와 다른점은 dtype의 설정이 가능하므로 같은 크기의 다른 종류의 배열을 만들 수 있다.

``` python
c = np.ones((2, 3, 4), dtype='i')
print(c)
#[[[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]

# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]]

k = np.ones_like(c, dtype='float64')

cc = np.copy(c)
```



#### 차원 관리

- 차원 축소 연산(demension reduction 연산)
  - 행렬의 하나의 행에 있는 원소들을 하나의 데이터 집합으로 보고 그 집합의 평균을 구하면 각행에 대해 하나의 숫자가 나오게 되는데 이 경우 차원 축소가 된다. 이러한 연산을 차원 축소 연산이라 한다.
- Numpy에서의 차원 축소 연산 명령 또는 메서드
  - 최대/최소 : min, max, argmin, argmax
  - 통계 : sum, mean, median, std, var
  - boolean : all, any

``` python
x = np.array([1, 10, 100])
print(x.min())
# 1

# argmin : 최소값의 위치
# argmax : 최대값의 위치
print(x.argmin())
# 0

print(x.argmax())
# 2

# mean : 평균값
# median : 중간값, 여러개의 숫자들이 있을 때 크기순으로 정렬한 뒤 가장 가운데 위치한 숫자 반환
# 개수가 짝수개일 경우 중간의 좌우값의 평균값을 반환한다.
# std :
# var :
print(x.mean())
# 37.0

print(np.median(x))
# 10.0

# all : 모든 조건이 참일 경우
# any : 하나의 조건이라도 들어 맞을 때
print(np.all([True, True, False]))
# False

print(np.any([False, False, True]))
# True
```

- axis 속성을 부여하는 메서드들은 대부분 차원 축소 명령에 속한다.



- 정렬
  - sort명령이나 메서드를 사용하여 배열 안의 원소를 크기에 따라 정렬하여 새로운 배열을 만들 수 있다.
  - 2차원 이상인 경우에는 행이나 열을 각각 따로 정렬할 수 있는데, 이때 axis 속성을 사용하여 행과 열을 결정할 수 있다.

``` python
a = np.array([[4, 3, 5, 7],
             [1, 12, 11, 9],
              [2, 15, 1, 14]
             ])
print(np.sort(a))	# default 값이 axis=1 이다
# [[ 3  4  5  7]
#  [ 1  9 11 12]
#  [ 1  2 14 15]]

print(np.sort(a, axis=0))
# [[ 1  3  1  7]
#  [ 2 12  5  9]
#  [ 4 15 11 14]]

print(a.argsort())	# 자료를 정렬하는 것이 아닌 순서만 알고싶을 때 사용된다.
# [[1 0 2 3]
#  [0 3 2 1]
#  [2 0 3 1]]
```



- 배열 더하기(합성)

``` python
a = np.array([1, 2, 3])
b = np.array([3, 2, 3])
print(np.column_stack((a, b)))
# [[1 3]
#  [2 2]
#  [3 3]]

# column_stack() : 배열을 열 기준으로 합침, 3개 이상도 사용 가능하다.
```



- 배열 나누기(쪼개기)
  - `split = array_split`

``` python
x = np.arange(9.0)
print(x)
# [0. 1. 2. 3. 4. 5. 6. 7. 8.]

x1 = np.split(x, 3)
# x1 = np.array_split(x, 3)
print(x1)
# [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]

print(x1[1])
# [3. 4. 5.]

x2 = np.split(x, [3, 4])	# 인덱스 3위치에서 한번 자르고 4 위치에서 한번 잘라서 나누겠다는 의미. 이러한 형식으로 자신이 원하는대로 자를 수 있다.
print(x2)
# [array([0., 1., 2.]), array([3.]), array([4., 5., 6., 7., 8.])]
```



- 3차원 배열 나누기

``` python
y = np.arange(16).reshape(2, 2, 4)
print(y)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]]

#  [[ 8  9 10 11]
#   [12 13 14 15]]]

# dsplit() : 열 기준으로 나누기
print(np.dsplit(y, 2))
# [array([[[ 0,  1],
#         [ 4,  5]],

#        [[ 8,  9],
#         [12, 13]]]), array([[[ 2,  3],
#                              [ 6,  7]],

#                             [[10, 11],
#        						 [14, 15]]])]

# hsplit() : 행 기준으로 나누기
print(np.hsplit(y, 2))
# [array([[[ 0,  1,  2,  3]],

#        [[ 8,  9, 10, 11]]]), array([[[ 4,  5,  6,  7]],

#        							  [[12, 13, 14, 15]]])]

# vsplit() : 차원 기준으로 나누기(수직)
print(np.vsplit(y, 2))
# [array([[[0, 1, 2, 3],
#         [4, 5, 6, 7]]]), array([[[ 8,  9, 10, 11],
#						           [12, 13, 14, 15]]])]
```



- 반복 생성

``` python
a = np.array([0, 1, 2])
print(np.tile(a, 2))
# [0 1 2 0 1 2]

print(np.tile(a, (2, 2)))
# [[0 1 2 0 1 2]
#  [0 1 2 0 1 2]]

b = np.array([[1, 2], [3, 4]])
print(np.tile(b, 2))
# [[1 2 1 2]
#  [3 4 3 4]]

print(np.tile(b, (2, 2)))
# [[1 2 1 2]
#  [3 4 3 4]
#  [1 2 1 2]
#  [3 4 3 4]]
```

- 배열 반복

``` python
print(np.repeat(3, 4))	# 3을 4번 반복하는 배열 생성
# [3 3 3 3]

x = np.array([[1, 2], [3, 4]])
print(np.repeat(x, 2))
# [1 1 2 2 3 3 4 4]

print(np.repeat(x, 3, axis=0))
# [[1 2]
#  [1 2]
#  [1 2]
#  [3 4]
#  [3 4]
#  [3 4]]
```



- 배열 요소에 대한 추가 및 삭제

``` python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 11, 12, 13]])
print(arr)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 11 12 13]]
```



- delete

``` python
print(np.delete(arr, 1))
# [ 1  3  4  5  6  7  8  9 11 12 13]

print(np.delete(arr, 1, 0)) 	# axis값을 줘야 배열의 요소 하나가 아닌 행전체 혹은 열 전체의 삭제가 가능하다.
# [[ 1  2  3  4]
#  [ 9 11 12 13]]

print(np.delete(arr, 2, 1))
# [[ 1  2  4]
#  [ 5  6  8]
#  [ 9 11 13]]
```



- insert

```python
print(np.insert(arr, 1, 100))
# [  1 100   2   3   4   5   6   7   8   9  11  12  13]

print(np.insert(arr, 1, 100, axis = 1)) # delete와 같은 맥락으로 axis값이 주어져야 한다.
# [[  1 100   2   3   4]
#  [  5 100   6   7   8]
#  [  9 100  11  12  13]]

print(np.insert(arr, 1, 100, 0))
# [[  1   2   3   4]
#  [100 100 100 100]
#  [  5   6   7   8]
#  [  9  11  12  13]]

print(np.insert(arr, 1, [100, 200, 300], axis=1))
# [[  1 100   2   3   4]
#  [  5 200   6   7   8]
#  [  9 300  11  12  13]]
```



- append

``` python
print(np.append(arr, [[100, 101, 102, 103]], axis=0))	# 정확하게 2차원 배열에 추가할 때에는 2차원 형식으로 추가해줘야 한다
# [[  1   2   3   4]
#  [  5   6   7   8]
#  [  9  11  12  13]
#  [100 101 102 103]]
```



- resize

``` python
print(np.resize(arr, (3, 5)))		# resize를 통해 사이즈를 크게 만들면 다시 첫 값부터 들어가게 된다.
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 11]
#  [12 13  1  2  3]]

print(np.resize(arr, (2, 3)))		 # resize를 통해 사이즈를 작게 만들면 뒤의 값부터 없어진다.
# [[1 2 3]
#  [4 5 6]]
```



- trim_zeros : 좌우의 0을 제거

``` python
aa = np.array((0, 0, 0, 1, 2, 3, 0, 1, 2, 0, 0))
print(np.trim_zeros(aa))
# [1 2 3 0 1 2]

print(np.trim_zeros(aa, 'f'))		 # 속성으로 f를 주면 앞부분의 0만 제거되고 b를 주면 뒷부분의 0만 제거된다.
# [1 2 3 0 1 2 0 0]
```



- unique : 중복된 값을 하나로 만들어준다.

``` python
a = np.array([1, 1, 2, 1, 2, 2, 3, 3, 3])
print(np.unique(a))
# [1 2 3]

aa = np.array([[1, 1, 3, 2], [2, 3, 3, 1]])
print(np.unique(aa))
# [1 2 3]
```



- diag : 대각선 값 추출

``` python
x = np.arange(9).reshape((3, 3))
print(np.diag(x))
# [0 4 8]

print(np.diag(x, k=1))		# k값 설정을 통해 몇번째 대각선을 출력할건지 선택이 가능하다.
# [1 5]

print(np.diag(x, k=-1))
# [3 7]

# diag()를 활용한 배열 생성
c = diag([1, 2, 3])
print(c)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```



- 요소의 재정렬
- flip : 뒤집어 엎다

``` python
b = np.arange(8).reshape((2, 2, 2))
print(b)
# [[[0 1]
#   [2 3]]

#  [[4 5]
#   [6 7]]]

print(np.flip(b, 0))	# axis=0을 줌으로써 z축을 변경
# [[[4 5]
#   [6 7]]

#  [[0 1]
#   [2 3]]]

print(np.flip(b, 1))	# axis=0을 줌으로써 차원별 값 뒤집기
# [[[2 3]
#   [0 1]]

#  [[6 7]
#   [4 5]]]

print(np.flip(b))		# 속성을 부여하지 않음으로 통째로 뒤엎기
# [[[7 6]
#   [5 4]]

#  [[3 2]
#   [1 0]]]

# fliplr : 왼쪽 / 오른쪽 좌우 뒤집기
c = np.diag([1, 2, 3])
print(np.fliplr(c))
# [[0 0 1]
#  [0 2 0]
#  [3 0 0]]

# flipud : 배열을 위/아래 방향으로 뒤집기
print(np.flipud(c))
# [[0 0 3]
#  [0 2 0]
#  [1 0 0]]
```



- rot90 : 배열을 90도만큼 회전시킴
  - k : 몇번 움직일것인지를 선택하는 속성
  - axes : 회전 축 설정(0, 1) 또는 (1, 0) / 0에서 1방향으로 축 설정?

``` python
m = np.array([[1, 2], [3, 4]])
print(m)
# [[1 2]
#  [3 4]]

print(np.rot90(m, k=1))
# [[2 4]
#  [1 3]]

print(np.rot90(m, k=1, axes=(1, 0)))
# [[3 1]
#  [4 2]]

print(np.rot90(m, k=1, axes=(0, 1)))
# [[2 4]
#  [1 3]]
```



### Numpy 기술 통계(descriptive statistics)

- len() : 데이터의 개수(count) => `np.mean(x)`
- mean() : 평균(average, mean) / 통계 용어로는 샘플 평균이라 함 => `np.mean(x)`
- var() : 분산(variance) / 통계 용어로 샘플 분산 => `np.var(x)`
- std() : 표준 편차(standard deviation) / 수학 기호로는 s라고 표현된다 => `np.std(x)`
- max() : 최대값(maximum) => `np.max(x)`
- min() : 최소값(minimum) => `np.min(x)`
- median() : 중앙값(median) => `np.median(x)`
- percentile() : 사분위수(quartile)
  - `np.percentile(x, 0)` : 최소값
  - `np.percentile(x, 24)` : 1사분위수
  - `np.percentile(x, 50)` : 2사분위수
  - `np.percentile(x, 75)` : 3사분위수
  - `np.percentile(x, 100)` : 최대값



- 난수(random) 발생
- rand() : 0과 1 사이의 숫자르 무작위 추출

``` python
print(np.random.rand())
# 0.008257472536423283
```



- seed(씨앗값) 설정 : 겉보기에는 무작위 수처럼 보이지만 실제로는 컴퓨터가 처음 만들어질 때 생성된 셋팅값에 의해 일정한 값이 추출된다.

``` python
np.random.seed(0)
print(np.random.rand(5))
# [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]

print(np.random.rand(7))
#[0.64589411 0.43758721 0.891773   0.96366276 0.38344152 0.79172504 0.52889492]

np.random.seed(0)
print(np.random.rand(5))
# [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]
```



- 데이터의 순서 바꾸기
- shuffle : 데이터를 섞음

``` python
x = np.arange(10)
print(x)
# [0 1 2 3 4 5 6 7 8 9]

np.random.shuffle(x)
print(x)
# [3 1 8 7 9 0 6 4 2 5]
```

- shuffle은 배열 자체를 섞는것이기 때문에 print(np.random.shuffle(x))는 None값이 리턴된다.



- 데이터 샘플링 : 이미 있는 데이터 집합에서 무작위로 선택하는 것

- choice : np.random.choice(a, size, replace, p)

  - a : 원본 데이터, 정수이면 range(a)
  - size : 샘플 숫자

  - replace : True - 한번 선택한 데이터를 다시 선택할 수 있다. / False - 한번 선택한 데이터는 다시 선택할 수 없다.
  - p : 배열, 각 데이터가 선택될 수 있는 확률

``` python
x = np.random.choice(5, 5, replaece=True)
print(x)
# [0 1 1 0 1]

x1 = np.random.choice(10, 3, replace=False)
print(x1)
# [6 1 9]

x2 = np.random.choice(5, 10, p=[0.2, 0, 0, 0.3, 0.5])
print(x2)
# [3 4 0 4 0 4 4 3 3 4]
```



- randn() : 가우시안 표준 정규 분포

``` python 
print(np.random.randn(5))
# [ 2.26975462 -1.45436567  0.04575852 -0.18718385  1.53277921]
```



- randint(low, high, size) : 균열 분포의 정수 난수
  - high 값이 없으면 low와 0 사이의 숫자, high 값이 있으면 low~high 사이의 숫자를 출력한다
  - size는 난수의 개수를 의미

``` python
print(np.random.randint(10, size = 10))
# [4 9 8 1 1 7 9 9 3 6]

print(np.random.randint(1, 45, size=7))
# [12 19 28  1 15 36 13]
```



- 정수 데이터 카운팅
- unique : 데이터에서 중복된 값을 제거하고 중복되지 않는 값의 리스트를 출력
  - return-counts 속성 : True - 데이터의 갯수도 출력

``` python
a = np.unique([1, 1, 2, 2, 2, 3, 3, 3, 3], return_counts=True)
print(a)
# (array([1, 2, 3]), array([2, 3, 4], dtype=int64))

b = np.array(['a', 'a', 'b', 'b', 'b', 'c' ,'c', 'c', 'c', 'c'])
data, counts = np.unique(b, return_counts = True)		# 이런식으로 변수를 따로 넣으면 각각 넣을 수 있다.
print(data)
# ['a' 'b' 'c']

print(counts)
# [2 3 5]
```



- bincount : minlength 인수를 설정하여 사용하면 편리하다.
  - unique는 실제로 나온 숫자만 카운트하므로 나올 수 있는 숫자에 대한 카운트는 하지 않는다.
  - 하지만 bincount는 나올 수 있는 숫자의 카운트를 0으로 출력한다.

``` python
print(np.bincount([1, 1, 2, 2, 3, 4], minlength=6))
# [0 2 2 1 1 0]
```





