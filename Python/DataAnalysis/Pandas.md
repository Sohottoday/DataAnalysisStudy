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



- 존재하지 않는 값은 NaN이 출력된다.
- index 지정을 통해 원하는 index 설정도 가능하다.

```python
df2 = pd.DataFrame(data, columns=['year', 'city', 'pop', 'debt'], index=['one', 'two', 'three', 'four'])
print(df2)  
#       year city   pop debt
#one    2000   서울  4000  NaN
#two    2001   부산  2000  NaN
#three  2002   광주  1000  NaN
#four   2001   대구  1000  NaN
```



- 원하는 컬럼 내용만 따로 확인이 가능하다.

```python
print(df2['city'])
#one      서울
#two      부산
#three    광주
#four     대구
#Name: city, dtype: object
```



- 컬럼값만 가져오는 방법

```python
print(df2.columns)
#Index(['year', 'city', 'pop', 'debt'], dtype='object')
```



- 행 전체 값을 가져오는 방법 (기존에 ix 메서드 였으나 최근 없어지고 loc와 iloc가 기능을 대체)
  - row(행)의 위치를 접근할 때 사용하는 메서드(index 값을 통해 검색)
  - 색인을 name속성의 값으로 할당한다.

```python
print(df2.loc['one'])
#year    2000
#city      서울
#pop     4000
#debt     NaN
#Name: one, dtype: object
```



- dataframe에 값을 넣는 방법

```python
df2['debt'] = 1000
print(df2)
#       year city   pop  debt
#one    2000   서울  4000  1000
#two    2001   부산  2000  1000
#three  2002   광주  1000  1000
#four   2001   대구  1000  1000

df2['debt'] = np.arange(4.)
print(df2)
#       year city   pop  debt
#one    2000   서울  4000   0.0
#two    2001   부산  2000   1.0
#three  2002   광주  1000   2.0
#four   2001   대구  1000   3.0

 # Series 객체는 index가 붙는 데이터형식이므로 그냥 넣으면 dataframe과 매칭이 안되어 오류가 난다.
val = Series([1000, 2000, 3000, 4000], index=['one', 'two', 'three', 'four'])     
df2['debt'] = val
print(df2)
#       year city   pop  debt
#one    2000   서울  4000  1000
#two    2001   부산  2000  2000
#three  2002   광주  1000  3000
#four   2001   대구  1000  4000

# 이런식으로 인덱스를 지정하지 않고 넣을 수 있는데 지정하지 않은 인덱스의 해당 값은 NaN값이 뜬다.
val1 = Series([1000, 3000, 5000], index=['one', 'three', 'four'])
df2['debt'] = val1      
print(df2)
#       year city   pop    debt
#one    2000   서울  4000  1000.0
#two    2001   부산  2000     NaN
#three  2002   광주  1000  3000.0
#four   2001   대구  1000  5000.0

df2['aaa'] = df2.city =='서울'
print(df2)
#       year city   pop    debt    aaa
#one    2000   서울  4000  1000.0   True
#two    2001   부산  2000     NaN  False
#three  2002   광주  1000  3000.0  False
#four   2001   대구  1000  5000.0  False
```



- 컬럼 지우는 방법

```python
del df2['aaa']
print(df2)
#       year city   pop    debt
#one    2000   서울  4000  1000.0
#two    2001   부산  2000     NaN
#three  2002   광주  1000  3000.0
#four   2001   대구  1000  5000.0
```



- 딕셔너리 형식 안에 또 하나의 딕셔너리가 존재하는 경우의 dataframe

```python
data2 = {
    'seoul' : {2001 : 20, 2002 : 30},
    'busan' : {2000 : 10, 2001 : 200, 2002 : 300}
}

df3 = pd.DataFrame(data2)
print(df3)
#      seoul  busan
#2001   20.0    200
#2002   30.0    300
#2000    NaN     10

# 메인 딕셔너리의 key값은 컬럼명으로, 내부 딕셔너리 객체의 key값은 index로 나타난다.
```

- 위의 컬럼과 row 값을 바꾸고 싶다면 T를 추가하면 된다.

``` PYTHON
print(df3.T)
#        2001   2002  2000
#seoul   20.0   30.0   NaN
#busan  200.0  300.0  10.0
```

- 데이터프레임에서 values 속성은 저장된 데이터를 2차원 배열로 리턴한다.

``` python
print(df3.values)
#[[ 20. 200.]
# [ 30. 300.]
# [ nan  10.]]
```



#### 색인

- 색인(index) 객체
  - pandas의 색인 객체는 표형식의 데이터에서 각 행과 열에 대한 헤더(이름)와 다른 메타데이터(축의 이름)를 저장하는 객체
  - Series나 DataFrame 객체를 생성할 때 사용되는 배열이나 순차적인 이름은 내부적으로 색인으로 변환된다.

``` python
obj = Series(range(3), index=['a', 'b', 'c'])
print(obj)
#a    0
#b    1
#c    2
#dtype: int64

idx = obj.index

print(idx)
# Index(['a', 'b', 'c'], dtype='object')
print(idx[1])
# b
print(idx[1:])
# Index(['b', 'c'], dtype='object')
```



- 색인 객체는 변경할 수 없다
  - `idx[1] = 'd'` : 에러가 뜬다

``` python
index2 = pd.Index(np.arange(3))
print(index2)
# Int64Index([0, 1, 2], dtype='int64')
```



- 재색인(reindex) : 새로운 색인에 맞도록 객체를 새로 생성하는 기능
  - 객체가 없는 값은 NaN값으로 대체하며 index 자체를 바꾸는 것이 아닌 출력 순서가 바뀌는 것이다.

``` python
obj = Series([2.3, 4.3, -4.1, 3.5], index=['d', 'b', 'a', 'c'])
print(obj)
#d    2.3
#b    4.3
#a   -4.1
#c    3.5
#dtype: float64

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
print(obj2)
#a   -4.1
#b    4.3
#c    3.5
#d    2.3
#e    NaN
#dtype: float64
```



- NaN값 대신 어떠한 값으로 채우고 싶다면 fill_value 속성을 이용한다.

``` python
obj3 = obj.reindex(['a', 'b', 'c', 'c', 'e', 'f'], fill_value=0.0)
print(obj3)
#a   -4.1
#b    4.3
#c    3.5
#c    3.5
#e    0.0
#f    0.0
#dtype: float64
```



- 종합 활용

``` python
df = DataFrame(np.arange(9).reshape(3, 3), index=['a', 'b', 'c'], columns=['x', 'y', 'z'])
print(df)
#   x  y  z
#a  0  1  2
#b  3  4  5
#c  6  7  8

df2 = df.reindex(['a', 'b', 'c', 'd'])
print(df2)
#     x    y    z
#a  0.0  1.0  2.0
#b  3.0  4.0  5.0
#c  6.0  7.0  8.0
#d  NaN  NaN  NaN

col = ['x', 'w', 'z']
print(df.reindex(columns = col))
#   x   w  z
#a  0 NaN  2
#b  3 NaN  5
#c  6 NaN  8
```



- mehtod 속성의 ffill을 활용해 앞의 값으로 채우는 방법도 있다.

```python
obj4 = Series(['blue', 'red', 'yellow'], index=[0, 2, 4])
print(obj4)
#0      blue
#2       red
#4    yellow
#dtype: object

obj5 = obj4.reindex(range(6), method='ffill')
print(obj5)
#0      blue
#1      blue
#2       red
#3       red
#4    yellow
#5    yellow
#dtype: object
```



- DataFrame 에서의 ffill
  - 데이터프레임에서 보간은 row(행)에 대해서만 이루어진다. 

``` python
df = DataFrame(np.arange(9).reshape(3, 3), index=['a', 'b', 'd'], columns=['x', 'y', 'z'])
col = ['x', 'y', 'w', 'z']
df3 = df.reindex(index=['a','b', 'c', 'd'], method = 'ffill', columns= col)
print(df3)
#   x  y   w  z
#a  0  1 NaN  2
#b  3  4 NaN  5
#c  3  4 NaN  5
#d  6  7 NaN  8
# 컬럼값은 NaN으로 채워지지 않으나 row값은 앞의 값으로 채워졌다.
```



#### 삭제

- Series 삭제
  - 여러개의 값을 지울 때에는 list형식으로 준다.

``` python
obj = Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
print(obj)
#a    0
#b    1
#c    2
#d    3
#e    4
#dtype: int32
    
obj2 = obj.drop('c')
print(obj2)
#a    0
#b    1
#d    3
#e    4
#dtype: int32

obj3 = obj.drop(['b', 'd', 'c'])
print(obj3)
#a    0
#e    4
#dtype: int32
```



- DataFrame 삭제
  - 컬럼을 지울때에는 axis 값을 1로 준다.

``` python
df = DataFrame(np.arange(16).reshape(4, 4), index = ['seoul', 'busan', 'daegu', 'incheon'], columns=['one', 'two', 'three', 'four'])
print(df)
#         one  two  three  four
#seoul      0    1      2     3
#busan      4    5      6     7
#daegu      8    9     10    11
#incheon   12   13     14    15

# 행을 지울때
new_df = df.drop(['seoul', 'busan'])
print(new_df)
#         one  two  three  four
#daegu      8    9     10    11
#incheon   12   13     14    15

# 컬럼을 지울때 => axis 값을 1로 준다.
new_df = df.drop(['one', 'three'], axis=1)
print(new_df)
#         two  four
#seoul      1     3
#busan      5     7
#daegu      9    11
#incheon   13    15
```



#### Series 슬라이싱

``` python
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj['b':'d'])
#b    1.0
#c    2.0
#d    3.0
#dtype: float64
```

- 슬라이싱을 통한 값 변경

``` python
obj['b' : 'c'] = 10
print(obj)
#a     0.0
#b    10.0
#c    10.0
#d     3.0
#dtype: float64
```



#### DataFrame 슬라이싱

``` python
data = DataFrame(np.arange(16).reshape(4, 4), index = ['seoul', 'busan', 'kwangju', 'daegu'],
columns = ['one', 'two', 'three', 'four'])
print(data)
#         one  two  three  four
#seoul      0    1      2     3
#busan      4    5      6     7
#kwangju    8    9     10    11
#daegu     12   13     14    15

print(data['two'])
#seoul       1
#busan       5
#kwangju     9
#daegu      13
#Name: two, dtype: int32

print(data[['one' , 'three']])
#         one  three
#seoul      0      2
#busan      4      6
#kwangju    8     10
#daegu     12     14

print(data[:2])
#       one  two  three  four
#seoul    0    1      2     3
#busan    4    5      6     7

print(data[2:])
#         one  two  three  four
#kwangju    8    9     10    11
#daegu     12   13     14    15

print(data[data['three'] > 7])
#         one  two  three  four
#kwangju    8    9     10    11
#daegu     12   13     14    15

print(data < 5)
#           one    two  three   four
#seoul     True   True   True   True
#busan     True  False  False  False
#kwangju  False  False  False  False
#daegu    False  False  False  False

data[data < 5] = 0
print(data)
#         one  two  three  four
#seoul      0    0      0     0
#busan      0    5      6     7
#kwangju    8    9     10    11
#daegu     12   13     14    15
```

- loc 활용

``` python
print(data.loc['seoul'])
#one      0
#two      0
#three    0
#four     0
#Name: seoul, dtype: int32

print(data.loc['busan', ['two', 'three']])
#two      5
#three    6
#Name: busan, dtype: int32
```

- loc를 활용하면 순서를 마음대로 지정이 가능하다

``` python
print(data.loc[['daegu', 'kwangju'], ['three', 'two']])
#         three  two
#daegu       14   13
#kwangju     10    9
```



### pandas 연산

#### 색인이 다른 객체를 더하는 산술연산

- Series나 DataFrame나 같이 겹쳐있는 값이 있다면 연산하고 그 외에는 NaN과 연산하면 NaN이 되는 법칙에 의해 NaN이 된다.

``` python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

s1 = Series([5, 6, -1, 2], index=['a', 'c', 'd', 'e'])
s2 = Series([3, 4, -1, 2, 7], index=['a', 'c', 'e', 'f', 'g'])

print(s1 + s2)
#a     8.0
#c    10.0
#d     NaN
#e     1.0
#f     NaN
#g     NaN
#dtype: float64

df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list('bcd'), index=['seoul', 'busan', 'kwangju'])
df2 = DataFrame(np.arange(12).reshape(4, 3), columns=list('bde'), index=['incheon', 'seoul', 'busan', 'suwon'])

print(df1 + df2)
#           b   c     d   e
#busan    9.0 NaN  12.0 NaN
#incheon  NaN NaN   NaN NaN
#kwangju  NaN NaN   NaN NaN
#seoul    3.0 NaN   6.0 NaN
#suwon    NaN NaN   NaN NaN

df3 = DataFrame(np.arange(12).reshape(3, 4), columns=list('abcd'))
df4 = DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'))
print(df3 + df4)
#      a     b     c     d   e
#0   0.0   2.0   4.0   6.0 NaN
#1   9.0  11.0  13.0  15.0 NaN
#2  18.0  20.0  22.0  24.0 NaN
#3   NaN   NaN   NaN   NaN NaN

print(df3.add(df4, fill_value=0))
#      a     b     c     d     e
#0   0.0   2.0   4.0   6.0   4.0
#1   9.0  11.0  13.0  15.0   9.0
#2  18.0  20.0  22.0  24.0  14.0
#3  15.0  16.0  17.0  18.0  19.0
```

- fill_value 속성은 NaN값은 0으로 채우겠다는 의미

  결론적으로 df4의 값과 0이 더해진 값이 된다.



- DataFrame과 Series와의 연산은 Numpy의 브로드캐스팅과 유사하다.

```python
print(df3.reindex(columns = df4.columns, fill_value = 0))
#   a  b   c   d  e
#0  0  1   2   3  0
#1  4  5   6   7  0
#2  8  9  10  11  0

arr = np.arange(12,).reshape(3, 4)
print(arr)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]

print(arr[0])
# [0 1 2 3]

print(arr -arr[0])
#[[0 0 0 0]
# [4 4 4 4]
# [8 8 8 8]]


#0 1 2 3    -    0 1 2 3
#4 5 6 7
#8 9 10 11
```

- DataFrame과 Series의 연산

``` python
df = DataFrame(np.arange(12).reshape(4, 3), columns=list('bde'), index=['seoul', 'kwangju', 'daegu', 'incheon'])
print(df)
#         b   d   e
#seoul    0   1   2
#kwangju  3   4   5
#daegu    6   7   8
#incheon  9  10  11

s1 = df.iloc[0]
print(s1)
#b    0
#d    1
#e    2
#Name: seoul, dtype: int32

print(df-s1)
#         b  d  e
#seoul    0  0  0
#kwangju  3  3  3
#daegu    6  6  6
#incheon  9  9  9

# s1의 0, 1, 2의 값이 df의 b d e에 모두 계산된다.

s2 = Series(range(3), index=list('bef'))
print(s2)
#b    0
#e    1
#f    2
#dtype: int64

print(df + s2)
#           b   d     e   f
#seoul    0.0 NaN   3.0 NaN
#kwangju  3.0 NaN   6.0 NaN
#daegu    6.0 NaN   9.0 NaN
#incheon  9.0 NaN  12.0 NaN

s3 = df['d']
print(s3)
#seoul       1
#kwangju     4
#daegu       7
#incheon    10
#Name: d, dtype: int32

print(df + s3)
#          b   d  daegu   e  incheon  kwangju  seoul
#seoul   NaN NaN    NaN NaN      NaN      NaN    NaN
#kwangju NaN NaN    NaN NaN      NaN      NaN    NaN
#daegu   NaN NaN    NaN NaN      NaN      NaN    NaN
#incheon NaN NaN    NaN NaN      NaN      NaN    NaN

# index가 완전히 새롭게 추가되는 경우이기 때문에 모두 NaN값이 뜨는 결과가 나온다.

# 행에 대한 연산을 수행해야 할 경우에는 함수를 이용한다. (add, sub 등) axis값을 주면 된다.
print(df.add(s3, axis=0))
#          b   d   e
#seoul     1   2   3
#kwangju   7   8   9
#daegu    13  14  15
#incheon  19  20  21

print(df.sub(s3, axis=0))
#         b  d  e
#seoul   -1  0  1
#kwangju -1  0  1
#daegu   -1  0  1
#incheon -1  0  1
```



#### 함수 적용과 매핑

- 배열의 각 원소에 적용되는 함수를 유니버셜 함수라 한다.
- numpy.random 모듈에 있는 randn 함수는 임의의 정규분포 데이터를 생성한다.

```python
df = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['seoul', 'busan', 'daegu', 'incheon'])
print(df)
#                b         d         e
#seoul    0.168250 -0.703088  0.305677
#busan    0.208095 -0.301753  0.303408
#daegu    0.127047 -0.559360  1.138457
#incheon -0.655010 -2.191097 -0.718302

print(np.abs(df))		# abs : 절대값으로 변환하는 함수
#                b         d         e
#seoul    0.168250  0.703088  0.305677
#busan    0.208095  0.301753  0.303408
#daegu    0.127047  0.559360  1.138457
#incheon  0.655010  2.191097  0.718302

f = lambda x : x.max()-x.min()

print(df.apply(f))  # 행 중심으로 계산
#b    0.863105
#d    1.889344
#e    1.856759
#dtype: float64

print(df.apply(f, axis=1))	# 열 중심으로 계산
#seoul      1.008765
#busan      0.605161
#daegu      1.697818
#incheon    1.536087
#dtype: float64
```



- 함수 적용

``` python
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

print(df.apply(f))
#            b         d         e
#min -0.101865 -0.057921 -0.630277
#max  0.494251  1.782748  0.576976
```



- 데이터 프레임 객체에서 실수 값을 문자열 포맷으로 변환할 경우 applymap함수를 이용한다.

``` python
format = lambda x: '%.2f' % x       # x를 소수점 둘째자리까지면 표기한다는 의미
print(df.applymap(format))
#             b      d      e
#seoul     0.02   1.17   0.50
#busan     0.49   0.80  -0.63
#daegu    -0.10   1.78   0.58
#incheon   0.41  -0.06  -0.29
```

-  Series 객체에서 실수 값을 문자열 포맷으로 변환 할 경우 map 함수를 이용한다.

``` python
print(df['e'].map(format))
#seoul       0.50
#busan      -0.63
#daegu       0.58
#incheon    -0.29
#Name: e, dtype: object
```



#### 정렬과 순위

- 행의 색인이나 열의 색인 순으로 정렬

``` python
s1 = df['e'].map(format)

print(s1.sort_index())    # index순으로 정렬하겠다는 의미
#busan      -0.63
#daegu       0.58
#incheon    -0.29
#seoul       0.50
#Name: e, dtype: object

df2 = DataFrame(np.arange(8).reshape(2, 4), index=['three', 'one'], columns=['d','a','b','c'])
print(df2)
#       d  a  b  c
#three  0  1  2  3
#one    4  5  6  7

print(df2.sort_index()) # row를 기준으로 정렬
#       d  a  b  c
#one    4  5  6  7
#three  0  1  2  3

print(df2.sort_index(axis=1))   # 컬럼 순으로 정렬
#       a  b  c  d
#three  1  2  3  0
#one    5  6  7  4

```

- 데이터는 기본적으로 오름차순으로 정렬이 된다. 내림차순으로 정렬할 때에는 ascending=False 해준다.

``` python
print(df2.sort_index(axis=1, ascending=False))
#       d  c  b  a
#three  0  3  2  1
#one    4  7  6  5
```

- 객체를 값에 따라 정렬할 경우에는 sort_values 메서드를 사용한다.

```python
obj = Series([4, 7, -3, 1])
print(obj.sort_values())
#2   -3
#3    1
#0    4
#1    7
#dtype: int64
```

- 정렬을 할 때 비어있는 값은 정렬시 가장 마지막에 위치한다.

``` python
obj2 = Series([4, np.nan, 8, np.nan, -10, 2])
print(obj2)
#0     4.0
#1     NaN
#2     8.0
#3     NaN
#4   -10.0
#5     2.0
#dtype: float64

print(obj2.sort_values(0))
#4   -10.0
#5     2.0
#0     4.0
#2     8.0
#1     NaN
#3     NaN
#dtype: float64
```



- DataFrame에서 값을 기준으로 정렬하고자 할 때

``` python
frame = DataFrame({'b':[4, 7, -5, 2], 'a':[0, 1, 0, 1]})
print(frame)
#   b  a
#0  4  0
#1  7  1
#2 -5  0
#3  2  1

print(frame.sort_values(by='b'))     # by 속성에 정렬하고자 하는 컬럼명을 입력해준다.
#   b  a
#2 -5  0
#3  2  1
#0  4  0
#1  7  1

print(frame.sort_values(by=['a', 'b']))     # 이처럼 리스트형태로 여러값을 줄 수 있고 a로 먼저 정렬하고 b로 정렬 하겠다는 의미
#   b  a
#2 -5  0
#0  4  0
#3  2  1
#1  7  1
```



- 순위를 정하는 함수 : rank()

``` python
obj3 = Series([7, -2, 7, 4, 2, 0, 4])
print(obj3.rank())      # 아무 속성을 주지 않으면 동률일 경우 .5등이 나온다
#0    6.5
#1    1.0
#2    6.5
#3    4.5
#4    3.0
#5    2.0
#6    4.5
#dtype: float64

print(obj3.rank(method='first'))    # method='first'는 동률일 경우 데이터의 순서에 따라 순위를 메긴다는 의미
#0    6.0
#1    1.0
#2    7.0
#3    4.0
#4    3.0
#5    2.0
#6    5.0
#dtype: float64

print(obj3.rank(ascending=False, method='first'))
#0    1.0
#1    7.0
#2    2.0
#3    3.0
#4    5.0
#5    6.0
#6    4.0
#dtype: float64

print(obj3.rank(ascending=False, method='max'))     # 동률인 값은 뒷단계 기준으로 랭크를 묶어서 출력한다.
#0    2.0
#1    7.0
#2    2.0
#3    4.0
#4    5.0
#5    6.0
#6    4.0
#dtype: float64
```



- 중복 색인
  - Series 객체는 중복 색인이 있어도 같이 내보낸다.
  - 중복되는 색인값이 없을 경우에는 색인을 이용한 결과로 스칼라 값을 반환하고
  - 중복되는 색인값이 있을 경우에는 색인을 이용한 결과로 Series 객체를 반환한다.

``` python
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])

print(obj)
#a    0
#a    1
#b    2
#b    3
#c    4
#dtype: int64

print(obj['a'])
#a    0
#a    1
#dtype: int64

print(obj['c'])
# 4

```



- DataFrame에서도 중복색인을 허용한다.

``` python
df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
print(df.loc['b'])
#          0         1         2
#b  0.512300 -1.491978  0.022621
#b  1.233202  0.302777  0.036826
```



### 기술 통계 계산

- pandas는 일반적인  수학/통계 메서드를 가지고 있다.
- pandas의 메서드는 처음부터 누락된 데이터를 제외하도록 설계되어 있다.

```python
df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
print(df)
#    one  two
#a  1.40  NaN
#b  7.10 -4.5
#c   NaN  NaN
#d  0.75 -1.3
```



- sum() : 각 컬럼의 합을 더해서 Series 객체를 반환

``` python
print(df.sum())
#one    9.25
#two   -5.80
#dtype: float64

print(df.sum(axis=1))   # 각 행의 합을 반환
#a    1.40
#b    2.60
#c    0.00
#d   -0.55
#dtype: float64
```



- cumsum() : 누산(누적 합계) 메서드

```python
print(df.cumsum())
#    one  two
#a  1.40  NaN
#b  8.50 -4.5
#c   NaN  NaN
#d  9.25 -5.8
```



- 전체 행이나 컬럼의 값이 NA가 아니라면 NA값은 제외시키고 계산을 하는데
  - skipna 속성은 전체 행이나 컬럼의 값이 NA가 아니라도 제외시키지 않을 수 있다.
  - skipna의 기본값은 True

```python
print(df.sum(axis=1, skipna=False))
#a     NaN
#b    2.60
#c     NaN
#d   -0.55
#dtype: float64
```



- idxmin, idxmax와 같은 메서드는 최소, 최대값을 가지고 있는 색인 값 같은 간접 통계를 반환한다.

``` python
print(df.idxmax())
#one    b
#two    d
#dtype: object

print(df.idxmin())
#one    d
#two    b
#dtype: object
```



- unique() : 중복된 값을 하나로 묶음

``` python
s1 = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
print(s1.unique())
#['c' 'a' 'd' 'b']
```



- value_counts() : 값의 수를 계산(도수, 카운팅), 반환값은 Series 객체
  - 결과값이 내림차순으로 출력됨.

``` python
print(s1.value_counts())
#c    3
#a    3
#b    2
#d    1
#dtype: int64
```



- isin() : 어떤 값이 Series에 있는지 나타내는 메서드
  - boolean type(True, False)을 반환한다.

``` python
mask = s1.isin(['b', 'c']) 
print(mask)
#0     True
#1    False
#2    False
#3    False
#4    False
#5     True
#6     True
#7     True
#8     True
#dtype: bool

print(s1[mask]) # 이런식으로 원하는 값만 뽑아낼 수 있다.
#0    c
#5    b
#6    b
#7    c
#8    c
#dtype: object
```



- DataFrame에서는 value_counts가 바로 적용되지 못하기 때문에 apply함수를 활용해 각각 적용시킬 수 있도록 해야한다.
- apply : 한줄씩 함수를 적용하겠다 라는 의미

``` python
data = DataFrame({
    'Q1' : [1, 3, 4, 3, 4], 
    'Q2' : [2, 3, 1, 2, 3],
    'Q3' : [1, 5, 2, 4, 4]
    })

print(data)
#   Q1  Q2  Q3
#0   1   2   1
#1   3   3   5
#2   4   1   2
#3   3   2   4
#4   4   3   4

print(data.apply(pd.value_counts))  
#    Q1   Q2   Q3
#1  1.0  1.0  1.0
#2  NaN  2.0  1.0
#3  2.0  2.0  NaN
#4  2.0  NaN  2.0
#5  NaN  NaN  1.0
```



- fillna(0) -> na값을 0으로 채우겠다는 함수를 활용해 발전된 결과값 얻기

``` python
print(data.apply(pd.value_counts).fillna(0))
#    Q1   Q2   Q3
#1  1.0  1.0  1.0
#2  0.0  2.0  1.0
#3  2.0  2.0  0.0
#4  2.0  0.0  2.0
#5  0.0  0.0  1.0
```



- 누락된 데이터 처리(pandas의 설계 목표 중 하나는 누락된 데이터를 쉽게 처리하는 것이다.)
- pandas에서는 누락된 데이터를 실수든 아니든 모두 NaN(Not a Number)으로 취급한다.

``` python
stringData = Series(['aaa', 'bbbb', np.nan, 'ccccc'])
print(stringData)
#0      aaa
#1     bbbb
#2      NaN
#3    ccccc
#dtype: object
```



- 이러한 NaN의 값은 파이썬의 None값 Na와 같은 값으로 취급된다.
- isnull() : NaN값이 있으면 True로 반환한다.

``` python
print(stringData.isnull())
#0    False
#1    False
#2     True
#3    False
#dtype: bool

stringData[0] = None
print(stringData.isnull())
#0     True
#1    False
#2     True
#3    False
#dtype: bool
```



- 누락된 데이터 골라내기
  - dropna를 사용하는 것이 유용한 방법이며, 사용 결과값으로 Series객체를 반환
- dropna() : na값을 배제시킴

``` python
data = Series([1, NA, 3.4, NA, 8])
print(data.dropna())
#0    1.0
#2    3.4
#4    8.0
#dtype: float64
```

- boolean을 이용해 직접 계산한 후 가져오기

``` python
print(data.notnull())
#0     True
#1    False
#2     True
#3    False
#4     True
#dtype: bool

print(data[data.notnull()])
#0    1.0
#2    3.4
#4    8.0
#dtype: float64
```



- DataFrame에서 누락된 데이터 골라 내기
- dropna()는 기본적으로 NA값이 하나라도 있는 row(행) 자체를 제외시켜 버린다.

``` python
data = DataFrame([[1, 5.5, 3], [1, NA, NA], [NA, NA, NA], [NA, 3.3, 3]])
print(data)
#     0    1    2
#0  1.0  5.5  3.0
#1  1.0  NaN  NaN
#2  NaN  NaN  NaN
#3  NaN  3.3  3.0

print(data.dropna())
#     0    1    2
#0  1.0  5.5  3.0
```

- `how = 'all'` 옵션을 주면 모든 값이 NA인 행만 제외된다.

``` python
print(data.dropna(how='all'))
#     0    1    2
#0  1.0  5.5  3.0
#1  1.0  NaN  NaN
#3  NaN  3.3  3.0
```

- 열의 값이 모두 NA인 경우에만 지우고자 할 때에는 역시 axis 속성을 1로 주면 된다.

```python
data[4] = NA
print(data)
#     0    1    2   4
#0  1.0  5.5  3.0 NaN
#1  1.0  NaN  NaN NaN
#2  NaN  NaN  NaN NaN
#3  NaN  3.3  3.0 NaN

print(data.dropna(axis=1, how='all'))
#     0    1    2
#0  1.0  5.5  3.0
#1  1.0  NaN  NaN
#2  NaN  NaN  NaN
#3  NaN  3.3  3.0
```

- thresh 속성 : 개수를 지정하여 지정한 개수 이상의 value가 들어 있는 행을 가져온다.

``` python
data2 = DataFrame([[1, 2, 3, NA], [NA, 33, 11, NA], [11, NA, NA, NA], [43, NA, NA, NA]])
print(data2)
#      0     1     2   3
#0   1.0   2.0   3.0 NaN
#1   NaN  33.0  11.0 NaN
#2  11.0   NaN   NaN NaN
#3  43.0   NaN   NaN NaN

print(data2.dropna(thresh=2))	# 값의 개수가 2개 이상인 행만 가져온다는 의미
#     0     1     2   3
#0  1.0   2.0   3.0 NaN
#1  NaN  33.0  11.0 NaN
```



- 누락된 값 채우기
  - DataFrame에서는 누락된 데이터를 완벽하게 골라낼 수 없으므로 다른 데이터도 함께 버려지게 된다.
  - 이런 경우에는 fillna 메서드를 활용해 빈 곳을 채워주면 데이터의 손실을 막을 수 있다.

``` python
print(data2.fillna(0))
#      0     1     2    3
#0   1.0   2.0   3.0  0.0
#1   0.0  33.0  11.0  0.0
#2  11.0   0.0   0.0  0.0
#3  43.0   0.0   0.0  0.0
```

- fillna의 활용에 따라 각 컬럼마다 다른 값을 채워넣을 수 있다.

```python
print(data2.fillna({1:10, 3;30}))
#      0     1     2     3
#0   1.0   2.0   3.0  30.0
#1   NaN  33.0  11.0  30.0
#2  11.0  10.0   NaN  30.0
#3  43.0  10.0   NaN  30.0

print(data2.fillna(method='ffill'))	# method='ffill' 바로 앞의 값을 자신에게 적용
#      0     1     2   3
#0   1.0   2.0   3.0 NaN
#1   1.0  33.0  11.0 NaN
#2  11.0  33.0  11.0 NaN
#3  43.0  33.0  11.0 NaN

print(data2.fillna(method='ffill', limit=1))	# ffill로 전달되는 값이 한번만 전달되게 제한(limit)하라는 의미

data3 = Series([1, NA, 4, NA, 7])
print(data3.fillna(data3.mean()))	# 평균으로 채우겠다는 의미
```



### 다중 색인

- 색인의 계층 : pandas의 중요 기능 중 하나

  ​						다중 색인 단계를 지정할 수 있다.

``` python
data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 1, 2]])
print(data)
# a  1   -0.387349
#    2   -0.133219
#    3   -0.424211
# b  1    1.412886
#    2   -0.877219
#    3   -2.051215
# c  1    0.379040
#    2   -2.825059
# d  1   -0.128052
#    2    1.286932
# dtype: float64

print(data.index)
#MultiIndex([('a', 1),
#            ('a', 2),
#            ('a', 3),
#            ('b', 1),
#            ('b', 2),
#            ('b', 3),
#            ('c', 1),
#            ('c', 2),
#            ('d', 1),
#            ('d', 2)],
#           )
```

- 다중 색인에 접근하기

``` python
print(data['b'])
# 1    1.412886
# 2   -0.877219
# 3   -2.051215
# dtype: float64

print(data['a':'c'])
# a  1   -0.387349
#    2   -0.133219
#    3   -0.424211
# b  1    1.412886
#    2   -0.877219
#    3   -2.051215
# c  1    0.379040
#    2   -2.825059
# dtype: float64

print(data[['a', 'd']])     # 2차원 이상의 배열을 선택하여 불러오고자 할 때에는 대괄호에 신경써야 한다.
# a  1   -0.387349
#    2   -0.133219
#    3   -0.424211
# d  1   -0.128052
#    2    1.286932
# dtype: float64

print(data[:, 2])       # 콜론으로 첫번째 상위계층은 제외하고 두번째 해당 값을 가져오라는 의미/ a,b,c,d 의 층에 존재하는 2번 값을 가져오라는 의미
# a   -0.133219
# b   -0.877219
# c   -2.825059
# d    1.286932
# dtype: float64
```

- DataFrame에서의 다중색인

``` python
df = DataFrame(np.arange(12).reshape(4, 3), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['seoul', 'busan', 'kwangu'], ['red', 'green', 'blue']])
print(df)
#     seoul busan kwangu
#       red green   blue
# a 1     0     1      2
#   2     3     4      5
# b 1     6     7      8
#   2     9    10     11
```

- 다중 색인으 ㅣ이름을 표시하고 싶을때

``` python
df.columns.names=['city', 'color']
df.index.names=['key1', 'key2']
print(df)
city      seoul busan kwangu
color       red green   blue
key1 key2
# a    1        0     1      2
#      2        3     4      5
# b    1        6     7      8
#      2        9    10     11

print(df['seoul'])
# color      red
# key1 key2
# a    1       0
#      2       3
# b    1       6
#      2       9
```



