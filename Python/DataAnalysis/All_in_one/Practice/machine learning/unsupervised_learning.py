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



# 군집화(Clustering)
## 데이터를 줬을 때 알아서 몇가지 카테고리로 분류하는 것

## K-means Clustering
### 중심화된 몇가지 점(k)를 찍고 데이터들을 분속하여 점들이 조금씩 이동하며 주변의 데이터들의 중심점을 찾아감
### 보통 Euclidean Distance를 많이 활용한다.
"""
군집화에서 가장 대중적으로 사용되는 알고리즘
centroid라는 중점을 기준으로 가장 가까운 포인트들을 선택하는 군집화 기법
ex)
    스팸 문자 분류
    뉴스 기사 분류
"""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, max_iter=500)       # 3개의 그룹으로 분류하라는 의미, KMeans의 약점이다(처음 접하는 데이터의 경우 몇가지로 나눠야 할지 모르기 때문)
# max_iter 를 통해 여러번 작업하게 만들어 더 최적화된 값을 찾을 수 있다.
cluster_data = kmeans.fit_transform(df.loc[:, 'sepal length (cm)' : 'petal width (cm)'])
print(cluster_data[:5])
print(kmeans.labels_)

sns.countplot(kmeans.labels_)       # 아래의 실제 target값과 분류된 값들을 비교
plt.show()

sns.countplot(df['target'])
plt.show()


## DBSCAN(Density-based spatial clustering of applications with noise)
"""
밀도 기반 클러스터링
    - 밀도가 높은 부분을 클러스터링 하는 방식
    - 어느점을 기준으로 반경 x내에 점이 n개 이상 있으면 하나의 군집으로 인식하는 방식
    - KMeans 에서는 n_cluster의 갯수를 반드시 지정해주어야 하나, DBSCAN에서는 필요 없음
    - 기하학적인 clustering도 잘 찾아냄
"""
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=2)            # eps(epsilon) : maximum distanse 즉, 데이터의 최대 거리라고 보면 된다.
dbscan_data = dbscan.fit_predict(df.loc[:, 'sepal length (cm)' : 'petal width (cm)'])       # fit_transform이 아니라 fit_predict다. 예상까지 함
print(dbscan_data)
# 군집화해야할 카테고리 숫자를 대략적으로만 알 때 eps와 min_samples를 조정하며 값을 찾아내면 된다.


## 실루엣 스코어(군집화 평가)
"""
클러스터링 품질을 정략적으로 평가해주는 지표
    1 : 클러스터링의 품질이 좋다.
    0 : 클러스터링의 품질이 안좋다(클러스터링의 의미 없음)
    음수 : 잘못 분류됨
"""
from sklearn.metrics import silhouette_samples, silhouette_score

score = silhouette_score(data_scaled, kmeans.labels_)
print(score)
samples = silhouette_samples(data_scaled, kmeans.labels_)
print(samples[:5])

# 위의 과정은 몇백개 혹은 천개 이상의 컬럼이 존재할 때 몇개의 컬럼으로 줄이는게 맞을지 등을 찾아낼 때 활용한다.

# [api 참고(scikit-learn 공식 도큐먼트)](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
def plot_silhouette(X, num_cluesters):
    for n_clusters in num_cluesters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    
        plt.show()


plot_silhouette(data_scaled, [2, 3, 4, 5])          # 앞은 데이터, 뒤는 군집화 하려는 컬럼 수
# 빨간 점선은 평균 실루엣 계수를 의미합니다. 즉, 평균
# 데이터 전체가 빨간 점선을 넘어가면 분류가 잘 되었다는 의미