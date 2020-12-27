import pandas as pd

df = pd.read_csv('seoul_house_price.csv')

print(df.head())

df = df.rename(columns={'분양가격(㎡)' : '분양가격'})
print(df.head())

# 1. 빈 값과 data type 확인하기
print(df.info())

# 2. 통계값 확인
print(df.describe())

# 3. 분양가격 column을 int 타입으로 변환
# df['분양가격'].astype(int) 를 했을 때 뜨는 에러를 하나하나 추적해가며 문제를 해결한다.
df['분양가격'] = df['분양가격'].str.strip()     # 공백 제거
df.loc[df['분양가격'] == '', '분양가격'] = 0           # 비어있는 분양가격 column에 0을 넣어준다.
df['분양가격'] = df['분양가격'].fillna(0)           # NaN값에도 0을 넣어준다.
df['분양가격'] = df['분양가격'].str.replace(',', '')        # 6,657 과 같이 , 가 있을 경우 제거해 준다.
df['분양가격'] = df['분양가격'].fillna(0)           # 전처리 과정 중 생긴 NaN값을 다시 0으로 대체해준다.
df['분양가격'] = df['분양가격'].str.replace('-', '')
df['분양가격'] = df['분양가격'].fillna(0)
df.loc[df['분양가격'] == '', '분양가격'] = 0

df['분양가격'] = df['분양가격'].astype(int)

print(df.info())            # 위 과정을 모두 거쳐 object 타입이었던 분양가격을 int로 변환시켰다.

# 규모구분 column에 불필요한 '전용면적' 제거
df['규모구분'] = df['규모구분'].str.replace('전용면적', '')


# 지역명 별로 평균 분양가격을 확인해 보자
print(df.groupby('지역명')['분양가격'].mean())

# 분양가격이 100보다 작은 행을 제거해보자(위에서 NaN 값이나 이상한 값은 0으로 대체했으므로 평균값에 영향을 주기 때문에 제거한다.)
idx = df.loc[df['분양가격']<100].index
df = df.drop(idx, axis=0)
print(df.count())           # 1485개로 줄어들었다.

# 전처리 후 지역명 별 평균 분양가격을 확인해 보자
print(df.groupby('지역명')['분양가격'].mean())

# 지역별 최고 비싼 분양가는?
print(df.groupby('지역명')['분양가격'].max())

# 연도별 분양 가격
print(df.groupby('연도')['분양가격'].mean())

# 피벗 테이블로 표현해보기
## 행 인덱스 : 연도
## 열 인덱스 : 규모구분
## 값 : 분양가
print(pd.pivot_table(df, index='연도', columns='규모구분', values='분양가격'))

# 연도별, 규모별 가격 알아보기(multi-index)
print(df.groupby(['연도', '규모구분'])['분양가격'].mean())      # 보기 불편하다
# 보기 편하도록 DataFrame 형식으로 변환
print(pd.DataFrame(df.groupby(['연도', '규모구분'])['분양가격'].mean()))        # 오와 열이 맞게 보여준다.

