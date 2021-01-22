# 비지도 학습
"""
비지도 학습(Unsupervised Learning)은 기계학습의 일종으로, 데이터가 어떻게 구성되었는지를 알아내는 문제의 범주에 속한다.
이 방법은 지도학습(supervised learning) 혹은 강화 학습(Reinforcement Learning)과는 달리 입력값에 대한 목표치가 주어지지 않는다.
    - 차원 축소 : PCA, LDA, SVD
    - 군집화 : KMeans Clustering, DBSCAN
    - 군집화 평가
"""


# 차원 축소
"""
feature의 갯수를 줄이는 것을 뛰어 넘어, 특징을 추출하는 역할을 하기도 함
계산 비용을 감소하는 효과
전반적인 데이터에 대한 이해도를 높이는 효과

ex) 데이터의 column이 1천개가 넘는다고 가정했을 때
컬럼의 개수가 너무 많으므로 머신러닝 성능이 제대로 나오지 않음
이럴때 보통 3가지 방법으로 처리하는데 -> 데이터 전처리, feature selection으로 데이터 분석을 통해 중요한 컬럼만 뽑아내어 학습
세번째로 차원 축소를 사용한다.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
data = iris['data']
df = pd.DataFrame(data, columns = iris['feature_names'])
df['target'] = iris['target']


## PCA 차원축소
"""
주성분 분석(PCA)는 선형 차원 축소 기법. 매우 인기있게 사용되는 차원 축소 기법중 하나
주요 특징 중 하나는 분산(variance)을 최대한 보존한다는 점
    components에 1보다 작은 값을 넣으면, 분산을 기준으로 차원 축소
    components에 1보다 큰 값을 넣으면, 해당 값을 기준으로 feature을 축소
"""

pca = PCA(n_components=2)           # feature를 2개만큼 축소시키라는 의미

data_scaled = StandardScaler().fit_transform(df.loc[:, 'sepal length (cm)' : 'petal width (cm)'])      # 보통 PCA를 사용하기 전에 StandardScaler로 스케일을 맞춰주는 작업이 필요하다.
pca_data = pca.fit_transform(data_scaled)
print(pca_data[:5])


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['target'])
plt.show()


from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=0.99)       # 분산을 유지하며 축소시키라는 의미 / 분산을 유지하다보니 3개의 컬럼으로 축소되었다.

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')

# sample_size = 50
# ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], alpha=0.6, c=df['target'])
# plt.savefig('./tmp.svg')
# plt.title("ax.plot")
# plt.show()


## LDA 차원 축소 : (Linear Discriminant Analysis) : 선형 판별 분석법(PCA와 유사)
### LDA는 클리스(class) 분리를 최대화하는 축을 찾기 위해 클래스 간 분산과 내부 분산의 비율을 최대화 하는 방식으로 차원을 축소한다.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
data_scaled = StandardScaler().fit_transform(df.loc[:, 'sepal length (cm)' : 'petal width (cm)'])       # LDA 역시 작업 전 스케일링해준다.
lda_data = lda.fit_transform(data_scaled, df['target'])     # 클래스 레이블을 뒤에 넣어줘야 분리할 수 있는 점을 찾는다.

print(lda_data[:5])
plt.scatter(lda_data[:, 0], lda_data[:, 1], c=df['target'])
plt.show()


## SVD(Singular Value Decomposition)
"""
상품의 추천 시스템에도 활용되어지는 알고리즘(추천시스템)
특이값 분해기법
PCA와 유사한 차원 축소 기법
scikit-learn 패키지에서는 truncated SVD (aka LSA)을 사용한다.
"""
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(data_scaled)




