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

