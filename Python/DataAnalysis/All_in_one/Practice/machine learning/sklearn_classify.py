import warnings

# 불필요한 경고 출력을 방지한다.
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 데이터 로드
from sklearn.datasets import load_iris

iris = load_iris()          # 데이터셋은 딕셔너리 형태로 되어있다.
"""
- DESCR : 데이터셋의 정보를 보여준다.
- feature_names : feature data의 컬럼 이름
- target : label data(수치형)
- target_names : label의 이름(문자형)
"""

print(iris['DESCR'])
data = iris['data']
feature_names = iris['feature_names']
print(feature_names)        # sepal : 꽃 받침, petal : 꽃잎
target = iris['target']

# 데이터프레임 만들기
df_iris = pd.DataFrame(data, columns=feature_names)
print(df_iris.head())

df_iris['target'] = target

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot('sepal width (cm)', 'sepal length (cm)', hue='target', palette='muted', data=df_iris)
plt.title('Sepal')
plt.show()

# 훈련
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'])        # split 뒤 첫번째는 feature data, 두번째는 target data
print("훈련 : ",x_train.shape, y_train.shape)
print("valid : ", x_valid.shape, y_valid.shape)
## 데이터를 분배하여 학습시킬 때 종종 클래스들의 분포가 불균형적인 경우가 있는데
## 이러한 경우에는 stratify : label로 균등하게 분배해준다.
x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])


# Logistic Regression
"""
# 로지스틱 회귀 : 1958년 제안한 확률 모델
    - 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법
    - Logistic Regression, 서포트 벡터 머신(SVM)과 같은 알고리즘은 이진 분류만 가능하다(2개의 클래스 판별만 가능)
      하지만, 3개 이상의 클래스에 대한 판별을 진행하는 경우, 다음과 같은 전략으로 판별한다.
        one-vs-rest(OvR) : k개의 클래스가 존재할 때, 1개의 클래스를 제외한 다른 클래스를 k개 만들어, 각각의 이진 분류에 대한 확률을 구하고 총합을 통해 최종 클래스를 판별
        one-vs-one(OvO) : 4개의 계절을 구분하는 클래스가 존재한다고 가정했을 때, 0 vs 1, 0 vs 2, 0 vs 3, ... , 2 vs 3 까지 nX(n-1)/2 개의 분류를 만들어 가장 많이 양성으로 선택된 클래스를 판별
        대부분 OvsR 전략을 선호한다.
"""
from sklearn.linear_model import LogisticRegression

## step 1. 모델 선언
model = LogisticRegression()

## step 2. 모델 학습
model.fit(x_train, y_train)

## step 3. 예측
prediction = model.predict(x_valid)
print(prediction[:5])

## 평가
print("LogisticcRegression 예측 값 : ", (prediction==y_valid).mean())            # 1.0이 출력되는데 이 말은 100%라는 의미


# SGDClassifier
"""
stochastic gradient descent(SGD) : 확률적 경사 하강법
"""
from sklearn.linear_model import SGDClassifier

## step 1. 모델 선언
sgd = SGDClassifier()

## step 2. 모델 학습
sgd.fit(x_train, y_train)

## step 3. 예측
prediction = sgd.predict(x_valid)

## 평가
print("SGD 예측 값 : ", (prediction==y_valid).mean())


# 하이퍼 파라미터(hyper parameter) 튜닝
"""
각 알고리즘 별, hyper-parameter의 종류가 다양하다.
모두 다 외워서 할 수는 없으므로 문서를 보고 적절한 가설을 세운 다음 적용하면서 검증해야 한다.
- random_state : 하이퍼 파라미터 튜닝시, 고정할 것 => 운이 좋아 비슷한 값만 훈련될 가능성도 존재하기 때문
- n_jobs = -1 : CPU를 모두 사용(학습속도가 빠름)
"""
sgd = SGDClassifier(penalty='elasticnet', random_state=0, n_jobs=-1)        # penalty : 오버피팅을 방지하는 옵션 - 자세한 부분은 document 확인
sgd.fit(x_train, y_train)
prediction = sgd.predict(x_valid)
print("하이퍼 파라미터 적용 예측 값 : ", (prediction==y_valid).mean())      # random_state로 인해 값이 달라진다.


# KNeighborsClassifier
"""
최근접 이웃 알고리즘
알고리즘을 사용할 때 k를 지정하게 되는데, k를 지정한 갯수만큼 가까운 값들을 보여준다는 의미
k = 3 일 때 -> 3개까지 가장 가까운 값들
일반적으로 k에는 홀수를 적어준다. -> 짝수일 경우 동점인 경우가 발생할 가능성이 있기 때문
"""
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
knc_pred = knc.predict(x_valid)
print("KNeighborsClassifier 예측 값 : ", (knc_pred == y_valid).mean())

knc = KNeighborsClassifier(n_neighbors=9)
knc.fit(x_train, y_train)
knc_pred = knc.predict(x_valid)
print("KNeightborsClassifier 9개 예측 값 : ", (knc_pred == y_valid).mean())

