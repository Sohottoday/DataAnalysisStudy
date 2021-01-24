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
df.boxplot(column=['CMEDV'])


