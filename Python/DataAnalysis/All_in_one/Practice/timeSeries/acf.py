# 자기상관함수(ACF)

"""
외부 요인이 얼마나 지속되는지 검토
    ex) 마케팅 활동 효과가 몇주간 지속되는지?
    이번주 진행한 마케팅 활동이 얼마나 지속되는지?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 환율 데이터
df = pd.read_csv('DEXKOUS.csv', parse_dates=['DATE'], index_col='DATE')
df['DEXKOUS'].replace('.', '', inplace=True)
df['DEXKOUS'] = pd.to_numeric(df['DEXKOUS'])
df.fillna(method='ffill', inplace=True)

# 연도별
df.plot(figsize=(16, 5))
plt.show()

# 일 단위에서 주 단위로 변환
df_w = df.resample('W-Fri').last()

print(df_w.head())     # 7일 단위로 출력된다.

# 2017년, 2019년 데이터만 추출
df_2017 = df_w[df_w.index.year == 2017]
df_2019 = df_w[df_w.index.year == 2019]

df_2017.plot()
plt.show()

df_2019.plot()
plt.show()

plot_acf(df_2017)
plt.show()

plot_acf(df_2019)
plt.show()

"""
2017년에는 한번 충격이 오면 0.8 이하로 떨어지는데
2019년 같은 경우에는 충격이 와도 0.9대로 유지된다.
"""

# 첫번째 행 : 2017년 데이터의 원계열, ACF, PACF
# 두번째 행 : 2019년 데이터의 원계열, ACF, PACF
figure, axes = plt.subplots(2, 3, figsize=(16, 7))
axes[0, 0].plot(df_2017)
axes[0, 0].set_title('original series(2017')
axes[1, 0].plot(df_2019)
axes[1, 0].set_title('original series(2019')

plot_acf(df_2017, ax=axes[0, 1])
plot_acf(df_2019, ax=axes[1, 1])

plot_pacf(df_2017, ax=axes[0, 2])
plot_pacf(df_2019, ax=axes[1, 2])

plt.show()

"""
2019년 5월 즈음에 큰 폭으로 상승할 만한 큰 외부충격이 있었다.
pacf 그래프를 보면 한두가지만 큰 폭으로 튀어있는데 이것은 전형적인 AR 모형이다.

2017년에 비해 2019년은 외부 충격이 오래 지속되었다. 3~4주까지  => Autocorrelation의 충격으로 인한 변화량(2019년은 떨어지는 폭이 적음)
2017년에는 외부충격이 다음기에 0.75 남아있지만, 2019년에는 0.9가 남아있다. (persistency가 증가하고 있다.)
 => 가입자, 사용자 마케팅효과 분석
 => 주가지수, 환율 : 외부 충격이 얼마나 오래 지속되는가.

이러한 것을 바탕으로 예측 모형을 제작할 수 있다.
"""


