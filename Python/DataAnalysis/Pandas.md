# Pandas 

- 데이터 분석 기능을 제공하는 라이브러리.
- 예를 들면, CSV파일 등의 데이터를 읽고 원하는 데이터 형식으로 변환



### pandas 자료구조

#### Series

- Series는 일차원 배열 같은 자료구조

```python
import pandas as pd
from pandas import Series, DataFrame

obj = Series([3, 22, 34, 11])
print(obj)
#0     3
#1    22
#2    34
#3    11
#dtype: int64

# enumerate() 함수를 쓴 것 처럼 1차원 배열을 인덱스와 함께 출력해준다.(색인을 보여준다.)
print(obj.values)
# [ 3 22 34 11]
print(obj.index)
# RangeIndex(start=0, stop=4, step=1)

# 인덱스 설정도 가능하다
obj2 = Series([4, 5, 6, 2], index=['c', 'd', 'e', 'f'])
print(obj2)
#c    4
#d    5
#e    6
#f    2
#dtype: int64
print(obj2['c'])
# 4
print(obj2[['d', 'f','c']]) # 여러개의 인덱스를 지정할 때에는 리스트형식으로 불러준다.
#d    5
#f    2
#c    4
#dtype: int64
print(obj2 * 2)     # Series에 연산도 가능하다.
#c     8
#d    10
#e    12
#f     4
#dtype: int64
print('d' in obj2)
#True
```

- Series는 python의 dict 타입을 대신할 수 있다.

```python
data = {
    'kim' : 3400,
    'hong' : 2000,
    'kang' : 1000,
    'lee' : 2400
}

obj3 = Series(data)
print(obj3)     # 단 인덱스의 순서는 key값의 사전 순으로 들어가게 된다.
#kim     3400
#hong    2000
#kang    1000
#lee     2400
#dtype: int64

name = [
    'woo',
    'hong',
    'kang',
    'lee'
]

obj4 = Series(data, index = name)
print(obj4)     # woo 라는 키를 가진 value는 없으므로 NaN
#woo        NaN
#hong    2000.0
#kang    1000.0
#lee     2400.0
#dtype: float64

# 누락된 데이터를 찾을 때 사용하는 함수 : isnull, notnull
print(pd.isnull(obj4))
#woo      True
#hong    False
#kang    False
#lee     False
#dtype: bool
print(pd.notnull(obj4))
#woo     False
#hong     True
#kang     True
#lee      True
#dtype: bool
```



#### DataFrame



