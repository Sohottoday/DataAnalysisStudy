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
