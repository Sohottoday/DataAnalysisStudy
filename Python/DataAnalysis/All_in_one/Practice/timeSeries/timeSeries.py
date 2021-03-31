# 시계열 데이터
"""
시계열 데이터의 특징

시계열에서 반드시 고려해야 할 사항
원계열 = Trend + Cycle(계절요인 등) + 불규칙 term
"""

# Monthly Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

"""
호주 당뇨병 치료약(anti-diabetic) 월별 sales 데이터 사용
모든 회사에 있는 월별 매출, 가입자, 등 실적 데이터에 활용

데이터 불러오기. plottig

계절요인 분리. Trend, Seasonal, residual을 포함하는 테이블 생성.
Insight 도출
    1. 월평균 성장율
    2. Seasonal 요소 분석
    3. residual 증감 여부
"""

df = pd.read_csv('a10.csv', parse_dates=['date'], index_col='date')       # parse_dates = 지정한 컬럼을 date 타입으로 불러옴
print(df.head())

df.value.plot()
plt.show()

# 너무 오래된 데이터도 존재해 최신 데이터만 사용
df_new = df[df.index > '1999-12-31']

result = seasonal_decompose(df_new, model='additive', two_sided=False)
# seasonal_decompose : 
## tow_sided : 뒤의 데이터만?

result.plot()
plt.show()

# 하나의 데이터로 모아본다.
df_re = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
df_re.columns=['obs', 'trend', 'seasonal', 'resid']
df_re.dropna(inplace=True)
df_re['year'] = df_re.index.year
df_re.head()

plt.figure(figsize=(16, 6))
plt.plot(df_re.obs)
plt.plot(df_re.trend)
plt.plot(df_re.seasonal+df_re.trend)
plt.legend(['observed', 'trend', 'seasonal+trend'])
plt.show()
# 2006년 2007년에는 ordinary한 cycle에서 벗어나는 경향이 있다. (residual 크다. )

# trend
df_re.trend.pct_change()
## 트렌드가 얼마나 증가하고 감소했는지 를 표시해준다.

df_re.trend.pct_change().dropna().plot(kind='bar', figsize=(16, 5))
plt.show()

# 연월이 시분초까지 표현되어 지저분하므로 정리해준다.

ax = df_re.trend.pct_change().dropna().plot(kind='bar', figsize=(16, 5))
def get_date(date):
    return (str(date.year) + '-' + str(date.month))

get_date(df_re.index)
ax.set_xticklabels(list(map(lambda x : get_date(x), df_re.index)))
plt.show()

# residual : unexpected 값들
# 연도별로 봤을 경우(평균)
df_re.groupby('year')['resid'].mean().plot(kind='bar')
plt.show()

# 안정한 시계열
"""
- 시계열이 Stationary 하다" = 시계열 데이터가 미래에 똑같은 모양일 확률이 매우 높다
즉, 시계열이 안정적이지 않으면 현재의 패턴이 미래에 똑같이 재현되지 않으므로, 그대로 예측 기법 적용하면 안된다.

- 불안정한 시계열을 그대로 예측에 활용하는 경우
    설명력(R2(R스퀘어)) 90% 이상, 정확도 90% 이상 나옴
    그러나, Spurious regression(가성적 회귀), Overfitting 등의 문제 발생
    따라서, 원계열(ex 환율)보다는 차분데이터(ex 환율 증감) 사용

'(Augmented) Dickey Fuller Test'
    귀무가설(Null Hypothesis) : 원계열은 안정적이지 않다.
    p-value가 0.05보다 작으면, 귀무가설 기각. 즉, 안정적인 시계열
    p-value가 0.05보다 크면, 귀무가설 채택. 즉, 불안정한 시계열
 즉, 결과값에 대한 p-value는 정말 특별한 법칙을 따르는 것인지(p-value 작음), 우연히 이런 결과가 나왔는지(p-value 큼) 척도

 - 불안정한 시계열을 안정적인 시계열로 변경하는 가장 보편적인 방법은 성장률로 변환해서 예측하는 것이다.
    대부분의 시계열 예측은 '변화'를 예측한다.
Log difference도 성장율을 나타낸다.
"""

# 시계열 데이터(단변량), 이것만은 꼭 분석하자
"""
- 계졀 요인
    트렌드와 계절 요인 분리 및 도시
    트렌드, 계절요인, residual 변화에 따른 insight 도출

- 주기에 따른 특성
    일단위 -> 주단위 데이터 변환 시각화(노이즈 제거)
    구간별 변동성 증가/감소 여부
        Rolling / Resample 함수 활용
- 안정성
    과거 데이터로 미래 예측 가능한지 검토

- 자기상관함수(ACF)
    외부요인이 얼마나 지속되는지 검토
    ex)마케팅 활동효과가 몇주간 지속되는지?
    이번주 진행한 마케팅 활동이 얼마나 지속되는지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 환율 데이터

# daily 데이터
df = pd.read_csv('DEXKOUS.csv', parse_dates=['DATE'], index_col='DATE')
print(df.info())
print(df.isnull().sum())

print(df.head())

# 점만 찍혀있는 데이터 제거
df['DEXKOUS'].replace('.', '', inplace=True)
df['DEXKOUS'] = pd.to_numeric(df['DEXKOUS'])        # 숫자 형식으로 바꿔준다.
df['DEXKOUS'] = df['DEXKOUS'].fillna(method='ffill', inplace=True)
# 점만 찍혀있는 값을 빈값으로 바꿔준 뒤 앞에 있는 값으로 채워넣음
print(df.info())
print(df.head())
df['DEXKOUS'].plot(figsize=(10, 6))
plt.show()
# 일단위 데이터라 보기 힘들다.

# resample : 일별데이터 -> 주단위 데이터, 월단위 데이터로 변환하는 함수
df.resample('M').last()     # resample을 활용해 월단위로 바꿔주는데 그 값을 월말(last) 값으로 가져오라는 의미
df.resample('W-Fri').last()     # 이러한 식으로 쓰면 주차별로 가져오는데 기준을 금요일로 가져온다는 의미

df.resample('W-Fri').last().plot(figsize=(15, 6))
plt.show()
# 훨씬 보기 편해졌다.

# rolling : 이전 xx일에 대한 이동평균, 이동 합(sum) 을 산출할 때 사용한다.
df.rolling(10).mean()       # 이전 10일에 대한 평균을 표시하라는 의미
print(df.rolling(10).mean())

## 보통 표준편차를 구할 때 많이 사용한다.
print(df.rolling(30).std())     # 30일 이내 이동한 값의 표준편차

print(df.rolling(30).std().resample('M').mean())       # 매월 말 기준으로 30일 이내의 변동량에 대한 표준편차 평균

df.rolling(30).std().resample('M').mean().plot()
plt.show()
