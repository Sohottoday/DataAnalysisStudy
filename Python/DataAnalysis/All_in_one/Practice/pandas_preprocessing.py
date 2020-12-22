import pandas as pd

df = pd.read_csv('korean-idol.csv')
df2 = pd.read_csv('korean-idol-2.csv')

print(df.head())
test_df = df.copy()
print(test_df.head())

# 결측값을 채워주는 fillna
test_df['키'].fillna(-1)        # na 값을 -1로 채우라는 의미
# fillna는 사용한다 해서 곧바로 덮어씌우는 것이 아니므로 덮어씌우고 싶다면 inplace 속성을 True로 준다.
# test_df['키]'].fillna(-1, inplace=True)               # 이 방법도 사용 가능하지만 보통 변수에 대압하는 방식이 더 선호된다.
# test_df = test_df['키'].fillna(-1)
print(test_df['키'].fillna(test_df['키'].mean()))           # na값을 평균값으로 채우겠다는 의미

# 빈 값(NaN)이 있는 행을 제거 dropna
## 별다른 조건 없이 dropna를 사용하는 경우 NaN값이 존재하는 행 전체를 제거한다.
## axis 조건을 통해 행/열을 드랍할 수 있다.
print(df.dropna())
print('*' * 30)
print(df.dropna(axis=0))        # 행을 드랍
print('*' * 30)
print(df.dropna(axis=1))        # 열을 드랍
print('*' * 30)

## how 속성
### 'any' :  한개라도 있는 경우 드랍
### 'all' : 모두 NaN인 경우 드랍
print(df.dropna(axis=0, how='any'))
print('*' * 30)

# 중복된 값 제거 : drop_duplicates
print(df['키'].drop_duplicates())
print('*' * 30)
## keep 속성으로 유지하고 싶은 데이터를 선택할 수 있다.('first' / 'last')
df['키'].drop_duplicates(keep='last')
## 행 전체 제거
print(df.drop_duplicates('그룹'))


# 행/열 제거하기 drop
## 행 제거
df.drop('그룹', axis=1)
df.drop(['그룹', '소속사'], axis=1)         # 여러 행을 제거할 때

## 열 제거
## row를 제거하고자 할 때 index와 axis=0을 준다.
df.drop(3, axis=0)
df.drop([3, 5], axis=0)

# DataFrame 합치기 : concat
## row 기준 합치기
## row에 합칠 때 pd.concat에 합칠 데이터 프레임을 list로 합쳐준다.
## row 기준으로 합칠 때 sort=False 옵션을 주어 순서가 유지되도록 한다.
df_concat = pd.concat([df, test_df], sort=False)        # 단, 단순히 이렇게 합칠 경우 index가 초기화 된게 아니라서 꼬이게 된다.
## reset_index()로 인덱스를 초기화해줄 수 있다.
df_concat.reset_index(drop=True)        # drop=True 를 통해 기존 합치기 전의 index들을 제거해준다.

## column 기준으로 합치기
## column 기준으로 합치고자 할 때 axis=1 옵션을 부여하면 된다.
df_concat2 = pd.concat([df, df2], axis=1)
print(df_concat2.head())
## 행의 개수가 맞지 않는 상태에서 column concat은 NaN 값으로 대체된다.

# DataFrame 병합하기 : merge
## concat은 row나 column 기준으로 단순하게 이어 붙이기
## merge는 특정 고유한 키(unique id) 값을 기준으로 병합
## df와 df2는 '이름'이라는 column이 겹친다. 따라서, '이름'을 기준으로 두 DataFrame을 병합할 수 있다.
## pd.merge(left, right, on='기준 column', how='left')
### left와 right는 병합할 두 dataframe을 대입, on 에는  병합의 기준이 되는 column을 넣어 준다. 
### how에는 left, right, inner, outer 라는 병합 방식 중 한가지를 선택
#### how에 left 옵션을 부여하면 left dataframe에 키 값이 존재하면 해당 데이터를 유지하고 병합한 right dataframe의 값은 NaN 값을 유지한다.
pd.merge(df, df2, on='이름', how='left')
#### 반대로 right 옵션을 부여하면 right dataframe을 기준으로 병합하게 된다.
#### 만약 left dataframe이 더 많은 데이터를 보유하고 있다면, right를 기준으로 병합하면 dataframe 사이즈가 줄어든다.
#### inner 방식은 두 dataframe에 모두 키 값이 존재하는 경우만 병합  (교집합 느낌)
#### outer 방식은 하나의 dataframe에 키 값이 존재하는 경우 모두 병합    (합집합 느낌) -> outer방식에서 없는 값은 NaN으로 대입
pd.merge(df, df2, on='이름', how='inner')

## column명은 다르지만, 동일한 성질의 데이터인 경우
### ex) 한 데이터는 이름으로 되어 있고 하나의 데이터는 성함으로 되있는 경우 결론적으로는 같은 결의 데이터
### 단순하게 on 속성이 아닌 left_on, right_on 으로 각각 지정해 줄 수 있다.

# pd.merge(df, df2, left_on='이름', right_on='성함', how='outer')





