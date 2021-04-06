
# ARMA : AR+MA(p, q)
"""
# ARMA(Auto Regressive Moving Average) 모델
- 예를들어, ARMA(2, 2) 모델은 AR(2) + MA(2) 모델임

# ARIMA(Auto Regressive Integrated Moving Average) 모델
- ARMA 모델의 원계열 Yt를 차분하여 Yt' 로 변환한 모형
- 예를 들어, ARIMA(2, 1, 2) 모델은 원계열을 1번 차분하고 AR(2) + MA(2)을 진행한 모델임

# ARIMA 모델을 적용할 때 주의점
- ARIMA(p, k, q)모델 : AR(p), Integrated(k), MA(q)
    라이브러리에 입력할 때 Yt를 입력하면, Stationary해질 때까지 Yt', Yt", ... Yt**k와 같이 차분
    차분하지 않는 경우, 설명력이 매우 높은 모형이 생성됨 -> Training 데이터에서만 정확도가 높은 잘못된 결과일 확률이 매우 높음
    시계열의 단순 차분값을 활용하기 보다 변동율로 변환하기 위해서 원계열에 log를 취하고나 △log(log difference)를 취하는 경우도 많음.
    원데이터        log(원데이터)       △log
    200            log(200)          NAN
    300            log(300)          log(300)-log(200)
    200            log(200)          log(200)-log(300)
    100            log(100)          log(100)-log(200)

# 예측 모형 만드는 순서
안정성 검토
    Yt가 안정적이지 않으면, △Yt가 안정적인지 확인 -> 보통 1~2번의 차분으로 안정적인 시계열이 됨(ARIMA(p, 1, q) 또는 ARIMA(p, 2, q) 선정)
데이터 특성에 맞는 모형 결정(AR차수와 MA차수)
    PACF peak와 ACF peak의 개수로 AR, MA계수 선정.
    PACF의 peak이 p개, ACF의 peak이 q개 이면, ARIMA(p, k, q) 모델 선정
학습
    특정 시점 이전 데이터(Training set)로 학습
평가
    특정 시점 이후 데이터(Test set)로 평가

- 다 지나간 학습데이터를 맞춰보는게 목적이 아니라면 테스트 데이터를 분리해서 사용해야 한다.
- AR 프로세스만 잘 학습해서 이전 관측치를 그 다음기에 그대로 예측하는 경향이 있다(딥러닝도 똑같음)
  '변동'을 예측하도록 보완하던지, 아예 Level보다는 변동을 예측(실무에서 주로 사용)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('DEXKOUS.csv', parse_dates=['DATE'], index_col='DATE')

df['DEXKOUS'].replace('.', '', inplace=True)
df['DEXKOUS'] = pd.to_numeric(df['DEXKOUS'])
df['DEXKOUS'].fillna(method='ffill', inplace=True)

print(df.info())

# 2019년 데이터만 수집
df = df[(df.index > '2019-01-01') & (df.index < '2020-01-01')]

# ARIMA(p, k, q) => k 결정
print(adfuller(df.DEXKOUS))
## p-value 가 0.36
## 귀무가설을 기각하지 못한다. = 데이터가 안정적이지 않다
## 따라서 한번 더 차분

print(adfuller(df.DEXKOUS.diff().dropna()))
## p-value가 0에 가까운 값이 출력된다.
## 안정적인 시계열 데이터가 되었다.
## k = 1 로 결정(1차 차분이 안정적이다.)

# ARIMA(p, k, q) => p, q 결정
## 2 x 3 subplot을 통해 그려본다.
figure, axes = plt.subplots(2, 3, figsize=(15, 7))

axes[0, 0].plot(df.DEXKOUS)
axes[1, 0].plot(df.DEXKOUS.diff())

axes[0, 0].set_title('original series')
axes[1, 0].set_title('1st difference series')

plot_acf(df.DEXKOUS, axes[0, 1])
plot_pacf(df.DEXKOUS, axes[0, 2])

plot_acf(df.DEXKOUS.diff(), axes[1, 1])
plot_pacf(df.DEXKOUS.diff(), axes[1, 2])

plt.tight_layout()
plt.show()

## AR 차수 : 3차 ~ 1차
## MA 차수 : 2차 ~ 0차

# ARIMA 예측 모델링
## ARIMA의 차수는 (3, 1, 2)
model = ARIMA(df.DEXKOUS, order=(3, 1, 2), freq='B')        # 환율 데이터는 토요일 일요일은 나오지 않으므로 제외한다는 의미 freq='B' / Business day만 설정한다는 의미
model_fit = model.fit(trend='nc')       # not constent 라는 의미 그 반대는 'c'
print(model_fit.summary())
"""
coef
ar.L1.D.DEXKOUS    -0.2878      
ar.L2.D.DEXKOUS    -0.9557      
ar.L3.D.DEXKOUS    -0.0885  
이 계수가 마이너스라는 의미는 환율이 한번 튀었을 때 다음기, 다다음기 3기 연속으로 줄어든다는 의미

P>|z|     [0.025      0.975]        # 0.025, 0.975는 신뢰구간이라는 의미
-----------------------------------------------------------------------------------
0.000      -0.413      -0.163
0.000      -1.014      -0.897
0.165      -0.214       0.037
0.000       0.175       0.239

p-value는 0에 가까울수록 좋다. 0.05보다 작아야 한다.
3번째 데이터는 p-value도 높고 신뢰구간도 마이너스 이므로 믿을 수 없는 데이터이다.

이를 통해 ARIMA 모델의 p값은 3이 아니라 2인것을 알아낼 수 있다.
"""

model = ARIMA(df.DEXKOUS, order=(2, 1, 2), freq='B')
model_fit = model.fit(trend='nc')
print(model_fit.summary())

# plot_predict의 함정
model_fit.plot_predict()
plt.show()
## 위의 그래프를 보고 예측과 실제 데이터가 거의 비슷했다! 라고 결론짓는 경우가 있는데 이것은 test, train 데이터를 분리한 뒤 예측한 것이 아니므로 오류가 있다.

# Training set, Test set을 나누어서 학습과 평가
## 시계열은 너무 오래된 데이터를 사용해서도 안되고 예측하려는 기간을 너무 길게 잡아도 안된다.
train = df.iloc[0:30]
test = df.iloc[30:35]        # 5일 정도만 예측

model = ARIMA(train, order=(2, 1, 2), freq='B')
model_fit = model.fit(trend='nc')
fc, se, conf = model_fit.forecast(5, alpha=0.05)       # alpha 는 신뢰구간이라는 의미 / fc=forecast, se=standard, conf=confidence

print('예측 환율 : ', fc)

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)      # 예측 환율 하한가
upper_series = pd.Series(conf[:, 1], index=test.index)      # 예측 환율 상한가

plt.figure(figsize=(12, 5))
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')

plt.fill_between(test.index, lower_series, upper_series, color='black', alpha=0.1)      # 신뢰구간 표시
plt.legend(loc='upper left')
plt.show()




