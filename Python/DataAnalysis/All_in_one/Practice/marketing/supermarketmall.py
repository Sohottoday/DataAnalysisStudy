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


# 데이터 분석
"""
# K-means를 사용한 클러스터링
K-means는 가장 빠르고 단순한 클러스터링 방법 중 한 가지 입니다.
scikit-learn의 cluster 서브패키지 KMeans 클래스를 사용한다.
- n_clusters : 군집의 갯수(default = 8)
- init : 초기화 방법 'random'이면 무작위, 'k-means++' 이면 K-평균++ 방법.(default=k-means++)
- n_init : centroid seed 시도 횟수. 무작위 중심위치 목록 중 가장 좋은 값을 선택한다.(default=10)
- max_iter : 최대 반복 횟수(default=300)
- random_state : 시드값(default=None)
"""

## Age & spending Score 두 가지 변수를 사용한 클러스터링
from sklearn.cluster import KMeans

### X1에 'Age', 'Spending Score (1-100)' 의 값을 넣어준다.
X1 = df[['Age', 'Spending Score (1-100)']].values
print(X1[:10])

### inertia(관성(응집도))라는 빈 리스트를 만들어준다.
inertia = []

### 군집수 n을 1에서 11까지 돌아가며 X1에 대해 k-means++ 알고리즘을 적용하여 inertia를 리스트에 저장한다.
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

print(inertia)
# 위 inertia 결과값은 n_clusters 값이 1, 2, 3, 4 등의 값일 때의 관성(응집도) 값
"""
# Inertia value를 이용한 적정 k 선택
- 관성(inertia)에 기반하여 n 개수를 선택한다.
- 관성(inertia) : 각 중심점(centroid)에서 군집 내 데이터간의 거리를 합산한 것으로 군집의 응집도를 나타낸다. 이 값이 작을수록 응집도가 높은 군집화이다.
    즉, 작을 수록 좋은 값
"""
plt.figure(1, figsize = (16, 5))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.8)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()
# 위 그래프를 보면 엘보 포인트가 4인것을 알 수 있다. 즉, k의 개수는 4개면 적절해 보인다.

# 군집수를 4로 지정하여 시각화해보기
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()

"""
연령-소비점수를 활용한 군집 4개는 아래와 같이 명명할 수 있다.
    저연령-고소비 군
    저연령-중소비 군
    고연령-중소비 군
    저소비 군
군집별 활용 전략 예시
    이 수퍼마켓mall의 경우 소비점수가 높은 고객들은 모두 40세 이하의 젊은 고객이다.
    소비점수가 높은 고객들은 연령대가 비슷한 만큼 비슷한 구매패턴과 취향을 가질 가능성이 높다.
    해당 군집의 소비자 특성을 더 분석해본 뒤 해당 군집의 소비자 대상 VIP 전략을 수립해본다.
    소비점수가 중간정도인 고객들에게는 연령에 따라 두 개 집단으로 나눠서 접근해본다.
    소비점수가 낮은 고객군은 연령대별로 중소비점수 군집에 편입될 수 있도록 접근해본다.
"""

# 'Annual Income (k$)', 'Spending Score (1-100)' 두 가지 변수를 사용한 클러스터링
## 앞서 예측했던 대로 소득과 소비점수를 활용해 결과를 도출해보자
# X2에 값을 넣어준다.
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# inertia 라는 빈 리스트를 만든다.
inertia = []

# 군집수 n을 1에서 11까지 돌아가며 X2에 대해 k-means++ 알고리즘을 적용하여 inertia를 리스트에 저장한다.
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n))
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)

plt.figure(1, figsize = (16, 5))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.8)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()

# 군집수를 5로 지정하여 시각화
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()
