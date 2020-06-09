# Numpy

- Numpy는 과학 계산을 위한 라이브러리로 다차원 배열을 처리하는데 필요한 여러 기능을 제공한다.
  - pip3 install numpy

## numpy배열

- numpy에서 배열은 동일한 타입의 값들을 갖는다. 

  배열의 차원을 rank라고 한다.

- shape : 각 차원의 크기를 튜플로 표시한 것

  - ex) 2행, 3열인 2차원 배열의 rank는 2이고, shape(2, 3)

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

  