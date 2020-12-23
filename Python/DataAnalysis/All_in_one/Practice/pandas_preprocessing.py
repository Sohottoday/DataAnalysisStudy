import pandas as pd
import numpy as np

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


# Series의 Type
## object : 일반 문자열 타입
## float : 실수
## int : 정수
## category : 카테고리
## datetime : 시간

## type 확인
print(df.info())
## type 변환하기 : astype
print(test_df['키'].dtypes)     # float64
aa = test_df['키'].fillna(-1)        # 결측치(NaN)이 존재하는 경우에는 type을 변환할 수 없으므로 fillna를 통해 임의의 값으로 채워준다.
aa = aa.astype(int)
print(aa)       # dtype이 int32로 변환되어 출력되는것을 확인할 수 있다.

## 날짜 변환하기(datetime 타입) : to_datetime 메서드
print(test_df['생년월일'])      # dtype이 object로 되어 있다.
pd.to_datetime(test_df['생년월일'])     # 이렇게 해준다고 해서 기존의 dataframe에는 반영되어 있지 않는다.
## 변환된 것을 df['날짜'] column에 다시 대입을 해줘야 정상적으로 변환된 값이 들어 간다.
test_df['생년월일'] = pd.to_datetime(test_df['생년월일'])
print(test_df['생년월일'])          # dtype: datetime64[ns] 로 표기되어 있다.
### datetime 타입으로 변환하는 이유는 매우 손쉽게 월, 일, 요일 등 날짜 정보를 세부적으로 추출해 낼 수 있다.
### datetime의 약어인 'dt'에는 다양한 정보들을 제공해 준다.
print(test_df['생년월일'].dt.year)
print(test_df['생년월일'].dt.day)       
# hour, minute, second, dayofweek(요일 : 0-월요일, 1-화요일, 2-수요일 ...), weekofyear(그 해의 몇번째 주인지)

# apply 함수 : Series나 Dataframe에 좀 더 구체적인 로직을 적용하고 싶은 경우 활용
## apply를 적용하기 위해서는 함수가 먼저 정의되어야 한다.
## apply는 정의한 로직 함수를 인자로 넘겨준다.
### ex) 목표 : 남자/여자의 문자열 데이터로 구성된 '성별' column을 1/0으로 바꿔보자
# test_df.loc[test_df['성별']=='남자','성별'] = 1
# test_df.loc[test_df['성별']=='여자','성별'] = 0
# print(test_df.head())           # 단 위의 방식은 단순히 남, 녀 2가지라 간단하지만 종류가 여러가지일 경우 조건을 모두 입력해야 하는 불편함이 있다.

## 먼저 apply에 활요할 함수를 정의한다.
def male_or_female(x):
    if x == '남자':
        return 1
    elif x == '여자':
        return 0
print(test_df['성별'].apply(male_or_female))
test_df['성별new'] = test_df['성별'].apply(male_or_female)
print(test_df.head())

### ex) 목표 : cm당 브랜드 평판지수를 구해보자(브랜드평판지수/키)
# apply에 활용할 함수 정의
def cm_to_brand(df):                # 브랜드 평판지수와 키 모두 필요하므로 dataframe 자체를 함수로 넘긴다.
    value = df['브랜드평판지수'] / df['키']
    return value
print(test_df.apply(cm_to_brand, axis=1))      # series를 넘길때에는 문제가 없지만 dataframe을 넘길때에는 행/열 어떻게 넘길지 지정해줘야 한다.
test_df['cm당 평판지수'] = test_df.apply(cm_to_brand, axis=1)
print(test_df.head())

# lambda 함수의 적용
f = test_df['성별'].apply(lambda x : 1 if x=='남자' else 0)
# f = lambda x : 1 if x =='남자' else 0
# test_df['성별'].apply(f)

## 실제로는 간단한 계산식을 적용하려는 경우에 많이 사용된다.
test_df['키'].apply(lambda x : x / 2)
test_df['키'].apply(lambda x : x ** 2)
## apply에 함수식을 만들어서 적용해주는 것과 동일하기 때문에 복잡한 조건식은 함수로, 간단한 계산식은 lambda로 적용할 수 있다.

# map : 값을 매핑!
my_map = {
    '남자' : 1,
    '여자' : 0
}       # dictonary 정의

test_df['성별'].map(my_map)     # dictionay 에 정의한 값대로 매핑하면 손쉽게 변경할 수 있다.


# 데이터프레임의 산술연산
df3 = pd.DataFrame({'통계' : [60, 70, 80, 85, 75], '미술' : [50, 55, 80, 100, 95], '체육' : [70, 65, 50, 95, 100]})
print(df3)
## column고 column간 연산
print(df3['통계'] + df3['미술'])
print(df3['통계'] - df3['미술'])
print(df3['통계'] * df3['미술'])
print(df3['통계'] / df3['미술'])
print(df3['통계'] % df3['미술'])

## 복합 연산
print(df3['통계'] + df3['미술'] + 10)
# 이를 통해 df3['통계미술합계] = df3['통계'] + df3['미술']          이런식으로 새 행을 만들 수 있다.
print(df3['통계'] + df3['미술'] - df3['체육'])
print(df3.mean(axis=1))

## mean(), sum()을 axis 기준으로 연산
print(df3.sum(axis=0))
print(df3.sum(axis=1))

## NaN 값이 존재할 경우 연산
## NaN 값이 하나라도 존재하는 경우 결과도 NaN으로 출력된다.

## DataFrame 과 DataFrame간의 연산
df4 = pd.DataFrame({'통계' : ['good', 'bad', 'ok', 'good', 'ok'], '미술' : [50, 55, 80, 100, 95], '체육' : [70, 65, 50, 95, 100]})
## 문자열이 포함된 dataframe의 경우
### 보통 에러가 발생한다. -> 문자열을 제거한 뒤 연산해 준다.

## column의 순서가 바껴있는 경우(df1은 미술, 통계 순서 / df2는 통계, 미술 순서일 경우)
### python에서 알아서 column이름을 맞춰서 연산한다.

## 행의 갯수가 다른 경우
### 행이 많은 dataframe 기준으로 출력하되 빈 값은 NaN으로 인지하여 연산한다.