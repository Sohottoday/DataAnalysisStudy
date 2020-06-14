import pandas as pd
from pandas import Series, DataFrame

obj = Series([3, 22, 34, 11])
print(obj)

# enumerate() 함수를 쓴 것 처럼 1차원 배열을 인덱스와 함께 출력해준다.(색인을 보여준다.)
print(obj.values)
print(obj.index)

# 인덱스 설정도 가능하다
obj2 = Series([4, 5, 6, 2], index=['c', 'd', 'e', 'f'])
print(obj2)
print(obj2['c'])
print(obj2[['d', 'f','c']]) # 여러개의 인덱스를 지정할 때에는 리스트형식으로 불러준다.
print(obj2 * 2)     # Series에 연산도 가능하다.
print('d' in obj2)

# Series는 python의 dict 타입을 대신할 수 있다.
data = {
    'kim' : 3400,
    'hong' : 2000,
    'kang' : 1000,
    'lee' : 2400
}

obj3 = Series(data)
print(obj3)     # 단 인덱스의 순서는 key값의 사전 순으로 들어가게 된다.

name = [
    'woo',
    'hong',
    'kang',
    'lee'
]

obj4 = Series(data, index = name)
print(obj4)     # woo 라는 키를 가진 value는 없으므로 NaN

# 누락된 데이터를 찾을 때 사용하는 함수 : isnull, notnull
print(pd.isnull(obj4))
print(pd.notnull(obj4))

data = {
    'Seoul' : 4000,
    'Busan' : 2000,
    'Incheon' : 1500,
    'Kwangju' : 1000
}
obj5 = Series(data)
print(obj5)

# 인덱스만 바꾸려고 할 때
cities = ['Seoul', 'Daegu', 'Incheon', 'Kwangju']
obj6 = Series(data, index=cities)
print(obj6)
print(obj5 + obj6)  # 서로 둘 다 존재하는 데이터만 더하여 출력해 준다. (NaN값과 일반값을 더하면 NaN이 됨)

# Series 객체와 Series의 색인(index)은 name이라는 속성이 존재한다.
obj6.name = '인구수'        # Series 객체의 이름
print(obj6)
obj6.index.name = '도시'
print(obj6)
obj6.index = ['Daejeon', 'Busan', 'jaeju', 'jeonju']
print(obj6)

# DataFrame
a = pd.DataFrame([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(a)


data = {
    'city' : ['서울', '부산', '광주', '대구'],
    'year' : [2000, 2001, 2002, 2001],
    'pop' : [4000, 2000, 1000, 1000]
}
df = pd.DataFrame(data)
print(df)

# 컬럼 순서를 원하는대로 지정할 수 있다.
df = DataFrame(data, columns = ['year', 'city', 'pop'])
print(df)


import numpy as np

data = {
    'city' : ['서울', '부산', '광주', '대구'],
    'year' : [2000, 2001, 2002, 2001],
    'pop' : [4000, 2000, 1000, 1000]
}
# 존재하지 않는 값은 NaN이 출력된다.
# index 지정을 통해 원하는 index 설정도 가능하다.
df2 = pd.DataFrame(data, columns=['year', 'city', 'pop', 'debt'], index=['one', 'two', 'three', 'four'])
print(df2)  

#원하는 컬럼 내용만 따로 확인이 가능하다.
print(df2['city'])

# 컬럼값만 가져오는 방법
print(df2.columns)

# 행 전체 값을 가져오는 방법    => ix 메서드 활용 : row(행)의 위치를 접근할 때 사용하는 메서드(index 값을 통해 검색)
print(df2.loc['one'])        # ix는 색인을 name속성의 값으로 할당한다.  // ix메서드가 최근 없어지고 loc와 iloc가 기능을 대체하고 있다.

# dataframe에 값을 넣는 방법
df2['debt'] = 1000
print(df2)

df2['debt'] = np.arange(4.)
print(df2)

val = Series([1000, 2000, 3000, 4000], index=['one', 'two', 'three', 'four'])      # Series 객체는 index가 붙는 데이터형식이므로 그냥 넣으면 dataframe과 매칭이 안되어 오류가 난다.
df2['debt'] = val
print(df2)

val1 = Series([1000, 3000, 5000], index=['one', 'three', 'four'])
df2['debt'] = val1      # 이런식으로 인덱스를 지정하지 않고 넣을 수 있는데 지정하지 않은 인덱스의 해당 값은 NaN값이 뜬다.
print(df2)

df2['aaa'] = df2.city =='서울'
print(df2)

# 컬럼 지우는 방법
del df2['aaa']
print(df2)

# 딕셔너리 형식 안에 또 하나의 딕셔너리가 존재하는 경우
data2 = {
    'seoul' : {2001 : 20, 2002 : 30},
    'busan' : {2000 : 10, 2001 : 200, 2002 : 300}
}

df3 = pd.DataFrame(data2)
print(df3)