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



- Series의 index 값을 바꾸려 할 때

```python
data = {
    'Seoul' : 4000,
    'Busan' : 2000,
    'Incheon' : 1500,
    'Kwangju' : 1000
}
obj5 = Series(data)
print(obj5)
#Seoul      4000
#Busan      2000
#Incheon    1500
#Kwangju    1000
#dtype: int64

# 인덱스만 바꾸려고 할 때
cities = ['Seoul', 'Daegu', 'Incheon', 'Kwangju']
obj6 = Series(data, index=cities)
print(obj6)
#Seoul      4000.0
#Daegu         NaN
#Incheon    1500.0
#Kwangju    1000.0
#dtype: float64
```

- Series끼리의 덧셈은 서로 둘 다 존재하는 데이터끼리 더하여 출력해 준다.
  - NaN값과 일반 값을 더하면 NaN이 된다.

```python
print(obj5 + obj6)
#Busan         NaN
#Daegu         NaN
#Incheon    3000.0
#Kwangju    2000.0
#Seoul      8000.0
#dtype: float64
```

- Series객체와 Series의 색인(index)은 name이라는 속성이 존재한다.

```python
obj6.name = '인구수'        # Series 객체의 이름
print(obj6)
#Seoul      4000.0
#Daegu         NaN
#Incheon    1500.0
#Kwangju    1000.0
#Name: 인구수, dtype: float64

obj6.index.name = '도시'
print(obj6)
#도시
#Seoul      4000.0
#Daegu         NaN
#Incheon    1500.0
#Kwangju    1000.0
#Name: 인구수, dtype: float64
```

- index 이름 변경도 가능하다.

```python
obj6.index = ['Daejeon', 'Busan', 'jaeju', 'jeonju']
print(obj6)
#Daejeon    4000.0
#Busan         NaN
#jaeju      1500.0
#jeonju     1000.0
#Name: 인구수, dtype: float64
```



#### DataFrame

- DataFrame은 2차원리스트(2차원 배열) 같은 자료구조
- R언의 data.frame과 비슷하다.

```python
import pandas as pd
a = pd.DataFrame([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(a)
#   0  1  2
#0  1  2  3
#1  4  5  6
#2  7  8  9
```

- dic(딕셔너리)형태의 데이터타입을 통해서도 dataframe 타입을 만들 수 있다.

```python
data = {
    'city' : ['서울', '부산', '광주', '대구'],
    'year' : [2000, 2001, 2002, 2001],
    'pop' : [4000, 2000, 1000, 1000]
}
df = pd.DataFrame(data)
print(df)
#  city  year   pop
#0   서울  2000  4000
#1   부산  2001  2000
#2   광주  2002  1000
#3   대구  2001  1000
```

- 컬럼 순서를 원하는대로 지정할 수 있다.

```python
df = DataFrame(data, columns = ['year', 'city', 'pop'])
print(df)
#   year city   pop
#0  2000   서울  4000
#1  2001   부산  2000
#2  2002   광주  1000
#3  2001   대구  1000
```



