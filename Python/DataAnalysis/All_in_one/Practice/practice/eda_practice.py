import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/BostonHousing2.csv")
print(df.head())

"""
Feature Description
    TOWN : 지역이름
    LON, LAT : 위도, 경도 정보
    CMEDV : 해당 지역의 집값(중간값)
    CRIM : 근방 범죄율
    ZN : 주택지 비율
    INDUS : 상업적 비즈니스에 활용되지 않는 농지 면적
    CHAS : 경계선이 강에 있는지 여부
    NOX : 산화 질소 농도
    RM : 자택당 평균 방 갯수
    AGE : 1940년 이전에 건설된 비율
    DIS : 5개의 보스턴 고용 센터와의 거리에 따른 가중치 부여
    RAD : radial 고속도로와의 접근성 지수
    TAX : 10000달러당 재산세
    PTRATIO : 지역별 학생-교사 비율
    B : 지역의 흑인 지수(1000(B-0.63)^2), B는 흑인의 비율
    LSTAT : 빈곤층의 비율
"""

# EDA(Exploratory Data Analysis : 탐색적 데이터 분석)
## 회귀 분석 종속(목표) 변수 탐색

print(df.shape)
print(df.isnull().sum())        # 결측치 확인
print(df.info())

### 'CMEDV' 피처 탐색
print(df['CMEDV'].describe())
df['CMEDV'].hist(bins=50)
plt.show()
df.boxplot(column=['CMEDV'])
plt.show()

## 회귀 분석 설명 변수 탐색
### 설명 변수들의 분포 탐색
numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
fig = plt.figure(figsize=(16, 20))
ax = fig.gca()

df[numerical_columns].hist(ax=ax)
plt.show()

### 설명 변수들의 상관관계 탐색
cols = ['CMEDV', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
corr = df[cols].corr(method='pearson')      # 피어슨 상관관계로 탐색
print(corr)
fig = plt.figure(figsize = (16, 20))
ax = fig.gca()
# 상관관계를 히트맵으로 표현
sns.set(font_scale=1.5)         # 스케일을 설정해주는 이유는 히트맵을 사용할 때 변수들의 폰트가 매우 작게 보일수도 있기 때문
hm = sns.heatmap(corr.values, annot=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols, ax=ax)        # fmt는 소수점 2번째 자리까지 출력해달라는 의미
plt.tight_layout()
plt.show()

### 설명 변수와 종속 변수의 관계 탐색 (상관관계 세부 탐색)
plt.plot('RM', 'CMEDV', data=df, linestyle='none', marker='o', markersize=5, color='blue', alpha=0.5)
plt.title('Scatter plot')
plt.xlabel("RM")
plt.ylabel("CM")
plt.show()      # 이를 통해 방이 클수록 집값이 비싸다는 것을 알 수 있다.

plt.plot('RM', 'LSTAT', data=df, linestyle='none', marker='o', markersize=5, color='blue', alpha=0.5)
plt.title('Scatter plot')
plt.xlabel("RM")
plt.ylabel("LSTAT")
plt.show()      # 이번 관계는 반비례 관계인것을 확인할 수 있다.

### 지역별 차이 탐색
fig = plt.figure(figsize=(12, 20))
ax = fig.gca()
sns.boxplot(x='CMEDV', y='TOWN', data=df, ax=ax)
plt.show()

### 범죄율 알아보기
fig = plt.figure(figsize=(12, 20))
ax = fig.gca()
sns.boxplot(x='CRIM', y='TOWN', data=df, ax=ax)
plt.show()
# 위와같은 탐색을 통해 인사이트를 찾아낼 수 있다.


## 집값 예측 분석 : 회귀분석
### 데이터 전처리
# 피처 표준화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scale_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df[scale_columns] = scaler.fit_transform(df[scale_columns])

# 데이터셋 분리
from sklearn.model_selection import train_test_split

x = df[scale_columns]
y = df['CMEDV']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

### 회귀 분석 모델 학습
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
print(lr.coef_)     # 학습이 잘 되었는지 확인

plt.rcParams['figure.figsize'] = [12, 16]
coefs = lr.coef_.tolist()       # 리스트 형태로 변환
coefs_series = pd.Series(coefs)     # 시리즈 형태로 변환

x_labels = scale_columns
ax = coefs_series.plot.barh()       # 객체 선언
ax.set_title('feature coef graph')
ax.set_xlabel('coef')
ax.set_ylabel('x_features')
plt.show()

### 학습 결과 해석
# R2 score, RMSE score 계산
# R2
print(model.score(x_train, y_train))
# 0.7490284664199387        이란 값이 출력되는데, 이는 저 수치만큼 문제를 잘 설명하고 있다는 의미(75점 정도가 나왔다는 의미)
print(model.score(x_test, y_test))
# 0.700934213532155         실제 테스트를 했을 때 70점 정도가 출력되었다는 의미(즉, 모의고사는 75점정도가 나왔고 실제 수능은 70점이 나왔다는 표현)

# RMSE
y_predictions = lr.predict(x_train)
print("train에 대한 정보 : ", sqrt(mean_squared_error(y_train, y_predictions)))

y_predictions = lr.predict(x_test)
print("test에 대한 정보 : ", sqrt(mean_squared_error(y_test, y_predictions)))

# 피처 유의성 검정
import statsmodels.api as sm            # ols모델로 했을때 더 디테일하게 결과를 알아볼 수 있다.

x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# 다중 공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
# 10을 기준점으로 하여 10보다 크면 해당 컬럼은 다른 피쳐와 굉장히 상관관계가 높아 다중 공선성을 발생시킨다는 의미
vif["VIF Factor"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
vif["feature"] = x_train.columns
print(vif.round(1))