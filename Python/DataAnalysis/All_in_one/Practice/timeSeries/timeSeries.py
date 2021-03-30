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
df_new = df[df.index > '1991-12-31']

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
plt.show()