# 미디어별 광고비와 세일즈 데이터를 가지고 최적의 마케팅 믹스를 구해본다.

"""
# 데이터 설명
    TV - TV 매체비
    radio - 라디오 매체비
    newspaper - 신문 매체비
    sales - 매출액

# 문제 정의
    - 전제
        실제로는 광고 매체비 이외의 많은 요인이 매출에 영향을 미친다.(영업인력 수, 입소문, 경기, 유행 등)
        이번 분석에서는 다른 요인이 모두 동일한 상황에서 매체비만 변경했을 때 매출액의 변화가 발생한 것이라고 간주해본다.
        실제로 Acquisition 단계에서는 종속변수가 매출액보다는 방문자수, 가입자수, DAU, MAU 등의 지표가 될 것이다.
        현재 2011년에 있다고 가정한다.
    - 분석의 목적
        각 미디어별로 매체비를 어떻게 쓰느냐에 따라서 매출액이 어떻게 달라질지 예측한다.
        궁극적으로는 매출액을 최대화할 수 있는 미디어 믹스의 구성을 도출한다.
        이 미디어믹스는 향후 미디어 플랜을 수립할 때 사용될 수 있다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Advertising.csv')
print(df.shape)
print(df.tail())
print(df.info())        # info는 결측치를 확인할 때 자주 사용한다.

## 분석에 필요한 컬럼만 선택한다.
df = df[['TV', 'radio', 'newspaper', 'sales']]

## 기술통계를 확인해본다.
print(df.describe())

## 변수간의 correlation을 확인한다.
print(df.corr())
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

## 변수간의 pairplot을 확인한다.
sns.pairplot(df[['TV', 'radio', 'newspaper', 'sales']])
plt.show()


