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

print(df3.T)


obj = Series(range(3), index=['a', 'b', 'c'])
print(obj)

idx = obj.index

print(idx)
print(idx[1])
print(idx[1:])

# 색인 객체는 변경할 수 없다.
# idx[1] = 'd'   => 에러가 뜬다.
index2 = pd.Index(np.arange(3))
print(index2)

# 재색인(reindex) : 새로운 색인에 맞도록 객체를 새로 생성하는 기능
obj = Series([2.3, 4.3, -4.1, 3.5], index=['d', 'b', 'a', 'c'])
print(obj)

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
print(obj2)
# 객체가 없는 값은 NaN값으로 대체하며 index 자체를 바꾸는 것이 아닌 출력 순서가 바뀌는 것이다.

# NaN값 대신 어떠한 값으로 채우고 싶다면 fill_value 속성을 이용한다.
obj3 = obj.reindex(['a', 'b', 'c', 'c', 'e', 'f'], fill_value=0.0)
print(obj3)

# 종합 활용
df = DataFrame(np.arange(9).reshape(3, 3), index=['a', 'b', 'c'], columns=['x', 'y', 'z'])
print(df)
df2 = df.reindex(['a', 'b', 'c', 'd'])
print(df2)
col = ['x', 'w', 'z']
print(df.reindex(columns = col))

# mehtod 속성의 ffill을 활용해 앞의 값으로 채우는 방법도 있다.
obj4 = Series(['blue', 'red', 'yellow'], index=[0, 2, 4])
print(obj4)
obj5 = obj4.reindex(range(6), method='ffill')
print(obj5)



df = DataFrame(np.arange(9).reshape(3, 3), index=['a', 'b', 'd'], columns=['x', 'y', 'z'])
col = ['x', 'y', 'w', 'z']
df3 = df.reindex(index=['a','b', 'c', 'd'], method = 'ffill', columns= col)
print(df3)      # 컬럼값은 NaN으로 채워지지 않으나 row값은 앞의 값으로 채워졌다.
# 데이터프레임에서 보간은 row(행)에 대해서만 이루어진다. 

# Series 삭제
obj = Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
print(obj)

obj2 = obj.drop('c')
print(obj2)
# 여러개의 값을 지울 때에는 list형식으로 준다.
obj3 = obj.drop(['b', 'd', 'c'])
print(obj3)

# DataFrame 삭제
df = DataFrame(np.arange(16).reshape(4, 4), index = ['seoul', 'busan', 'daegu', 'incheon'], columns=['one', 'two', 'three', 'four'])
print(df)

# 행을 지울때
new_df = df.drop(['seoul', 'busan'])
print(new_df)

# 컬럼을 지울때 => axis 값을 1로 준다.
new_df = df.drop(['one', 'three'], axis=1)
print(new_df)

obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj['b':'d'])

# 슬라이싱을 통한 값 변경
obj['b' : 'c'] = 10
print(obj)

data = DataFrame(np.arange(16).reshape(4, 4), index = ['seoul', 'busan', 'kwangju', 'daegu'],
columns = ['one', 'two', 'three', 'four'])
print(data)
print(data['two'])
print(data[['one' , 'three']])
print(data[:2])
print(data[2:])
print(data[data['three'] > 7])
print(data < 5)
data[data < 5] = 0
print(data)

# loc
print(data.loc['seoul'])
print(data.loc['busan', ['two', 'three']])
# 순서를 마음대로 지정 가능하다.
print(data.loc[['daegu', 'kwangju'], ['three', 'two']])



s1 = Series([5, 6, -1, 2], index=['a', 'c', 'd', 'e'])
s2 = Series([3, 4, -1, 2, 7], index=['a', 'c', 'e', 'f', 'g'])

print(s1 + s2)

df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list('bcd'), index=['seoul', 'busan', 'kwangju'])
df2 = DataFrame(np.arange(12).reshape(4, 3), columns=list('bde'), index=['incheon', 'seoul', 'busan', 'suwon'])

print(df1 + df2)
# Series나 DataFrame나 같이 겹쳐있는 값이 있다면 연산하고 그 외에는 NaN과 연산하면 NaN이 되는 법칙에 의해 NaN이 된다.

df3 = DataFrame(np.arange(12).reshape(3, 4), columns=list('abcd'))
df4 = DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'))
print(df3 + df4)

print(df3.add(df4, fill_value=0))
# fill_value 속성은 NaN값은 0으로 채우겠다는 의미
# 결론적으로 df4의 값과 0이 더해진 값이 된다.

# DataFrame과 Series간의 연산
## Numpy의 브로드캐스팅과 유사하다
print(df3.reindex(columns = df4.columns, fill_value = 0))

arr = np.arange(12,).reshape(3, 4)
print(arr)
print(arr[0])
print(arr -arr[0])
#0 1 2 3    -    0 1 2 3
#4 5 6 7
#8 9 10 11



df = DataFrame(np.arange(12).reshape(4, 3), columns=list('bde'), index=['seoul', 'kwangju', 'daegu', 'incheon'])
print(df)

s1 = df.iloc[0]
print(s1)

print(df-s1)
# s1의 0, 1, 2의 값이 df의 b d e에 모두 계산된다.

s2 = Series(range(3), index=list('bef'))
print(s2)

print(df + s2)

s3 = df['d']
print(s3)

print(df + s3)
# index가 완전히 새롭게 추가되는 경우이기 때문에 모두 NaN값이 뜨는 결과가 나온다.

# 행에 대한 연산을 수행해야 할 경우에는 함수를 이용한다. (add, sub 등) axis값을 주면 된다.
print(df.add(s3, axis=0))
print(df.sub(s3, axis=0))

# 함수 적용과 매핑
## 배열의 각 원소에 적용되는 함수를 유니버셜 함수라 한다.

# numpy.random 모듈에 있는 randn 함수는 임의의 정규분표 데이터를 생성한다.
df = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['seoul', 'busan', 'daegu', 'incheon'])
print(df)

print(np.abs(df))
#절대값으로 변환하는 함수

f = lambda x : x.max()-x.min()

print(df.apply(f))  # 행 중심으로 계산
print(df.apply(f, axis=1))



def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

print(df.apply(f))

# 데이터 프레임 객체에서 실수 값을 문자열 포맷으로 변환할 경우 applymap함수를 이용한다.
format = lambda x: '%.2f' % x       # x를 소수점 둘째자리까지면 표기한다는 의미
print(df.applymap(format))

# Series 객체에서 실수 값을 문자열 포맷으로 변환 할 경우 map 함수를 이용한다.

print(df['e'].map(format))

# 정렬과 순위
## 행의 색인이나 열의 색인 순으로 정렬
s1 = df['e'].map(format)

print(s1.sort_index())    # index순으로 정렬하겠다는 의미

df2 = DataFrame(np.arange(8).reshape(2, 4), index=['three', 'one'], columns=['d','a','b','c'])
print(df2)
print(df2.sort_index()) # row를 기준으로 정렬
print(df2.sort_index(axis=1))   # 컬럼 순으로 정렬

# 데이터는 기본적으로 오름차순으로 정렬이 된다. 내림차순으로 정렬할 때에는 ascending=False 해준다.
print(df2.sort_index(axis=1, ascending=False))

# 객체를 값에 따라 정렬할 경우에는 sort_values 메서드를 사용한다.
obj = Series([4, 7, -3, 1])
print(obj.sort_values())

# 정렬을 할 때 비어있는 값은 정렬시 가장 마지막에 위치한다.
obj2 = Series([4, np.nan, 8, np.nan, -10, 2])
print(obj2)
print(obj2.sort_values(0))



# 값을 기준으로 정렬
frame = DataFrame({'b':[4, 7, -5, 2], 'a':[0, 1, 0, 1]})
print(frame)

print(frame.sort_values(by='b'))     # by 속성에 정렬하고자 하는 컬럼명을 입력해준다.

print(frame.sort_values(by=['a', 'b']))     # 이처럼 리스트형태로 여러값을 줄 수 있고 a로 먼저 정렬하고 b로 정렬 하겠다는 의미

# 순위를 정하는 함수 : rank()
obj3 = Series([7, -2, 7, 4, 2, 0, 4])
print(obj3.rank())      # 아무 속성을 주지 않으면 동률일 경우 .5등이 나온다

print(obj3.rank(method='first'))    # method='first'는 동률일 경우 데이터의 순서에 따라 순위를 메긴다는 의미

print(obj3.rank(ascending=False, method='first'))

print(obj3.rank(ascending=False, method='max'))     # 동률인 값은 뒷단계 기준으로 랭크를 묶어서 출력한다.