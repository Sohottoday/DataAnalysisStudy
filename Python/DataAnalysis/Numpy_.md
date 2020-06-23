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
# 2
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







