# 고객 데이터 분석을 통한 고객 세그먼트 도출

"""
수퍼마켓 몰 고객 데이터 분석을 통해 고객 세그먼트를 도출하고 그 사용법을 고민하자

# 데이터 설명
CustomerID : 고객들에게 배정된 유니크한 고객 번호
Gender : 고객의 성별
Age : 고객의 나이
Annual Income(k$) : 고객의 연 소득
Spending Score(1-100) : 고객의 구매행위와 구매 특성을 바탕으로 mall에서 할당한 고객의 지불 점수

# 문제 정의
- 전제
    주어진 데이터가 적절 정확하게 수집, 계산된 것인지에 대한 검증부터 시작해야하지만, 지금은 주어진 데이터가 정확하다고 가정(ex)Spending Score)
    주어진 변수들을 가지고 고객 세그먼트를 도출
    가장 적절한 수의 고객 세그먼트를 도출
- 분석의 목적
    각 세그먼트 별 특성을 도출
    각 세그먼트별 특성에 맞는 활용방안, 전략을 고민
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
print(df.tail())
print(df.info())
print(df.describe())
print(df.corr())
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

sns.pairplot(df[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()
"""
샘플데이터라 그런지 scatter plot을 보는 것만으로도 세그먼트가 눈에 보이는 것 같다.
특히 Spending Score와 Annual Income 사이의 관계에 따라 5개의 세그먼트가 나눠지는 것 처럼 보인다.
각 세그먼트는 해석이 가능하다.
실제 데이터는 이렇게 예쁘게 나눠떨어지기 어렵다는 점을 감안하고 분석을 해야한다.
"""

# 각 변수의 분포 확인
## Age
sns.distplot(df['Age'])
plt.show()

## Annual Income (k$)
sns.distplot(df['Annual Income (k$)'])
plt.show()

## Spending Score (1-100)
sns.distplot(df['Spending Score (1-100)'])
plt.show()

## Gender
sns.countplot(data=df, x='Gender')
plt.show()

## 성별에 따른 분포도 확인
sns.lmplot(data=df, x='Age', y='Annual Income (k$)', hue='Gender', fit_reg=False)          # lmplot은 보통 수치형 데이터끼리의 관계를 비교할 때 사용된다.
plt.show()
### 이 결과 남녀에 따른 결과는 큰 의미가 없다는 것을 알 수 있다.

# 남녀에 따른 전체적인 비교
sns.pairplot(df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], hue='Gender')
plt.show()

# 성별에 따른 boxplot
sns.boxenplot(x='Gender', y='Age', hue='Gender', palette=['m', 'g'], data=df)
plt.show()
