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


# 서포트 벡터 머신(SVC)
"""
- 딥러닝이 나오기 전까지 굉장히 성능이 좋았던 알고리즘
- 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 이진 선형 분류 모델을 만듦
- 경계로 표현되는 데이터들 중 가장 큰 폭을 가진 경계를 찾는 알고리즘
- LogisticRegression과 같이 이진 분류만 가능하다(2개의 클래스 판별만 가능)
  OvR 전략 사용
"""
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_valid)

print("SVC 예측값 : ", (svc_pred == y_valid).mean())

## 각 클래스 별 확률값을 RETURN 해주는 decision_function()
print(svc_pred[:5])       # 선택된 클래스의 값
print(svc.decision_function(x_valid)[:5])     # 클래스가 선택된 이유(가장 높은 확률이 선택되므로)


# 의사 결정 나무(Decision Tree) : 스무고개처럼 나무 가지치기를 통해 소그룹으로 나누어 판별하는 것
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_pred = dtc.predict(x_valid)

print("의사결정트리 예측 값 : ", (dtc_pred==y_valid).mean())

# 트리 알고리즘의 시각화
"""
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image 

def graph_tree(model):
    # .dot 파일로 export 해준다.(내보내 준다.)
    export_graphviz(model, out_file='tree.dot')

    # 생성된 .dot 파일을 .png로 변환
    call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'])

    # .png 출력
    return Image(filename = 'decistion-tree.png', width=500)

graph_tree(dtc)

- gini 계수 : 불순도를 의미하며, 계수가 높을수록 엔트로피가 크다는 의미이며 
엔트로피가 크다는 의미는 쉽게 말해 클래스가 혼잡하게 섞여 있다는 뜻이다.

det = DecisionTreeClassifier(max_depth=2)
와 같이 max_depth를 통해 트리구조의 깊이를 제한할 수 있다.
데이터가 많을 경우 트리구조가 지나치게 깊어져 과적합 현상이 나타날 수 있기 때문
"""


# 오차(Error)
## 정확도의 함정
from sklearn.datasets import load_breast_cancer   # 유방암 환자 데이터셋

cancer = load_breast_cancer()
print(cancer['DESCR'])

data = cancer['data']
target = cancer['target']
feature_names = cancer['feature_names']

df = pd.DataFrame(data = data, columns = feature_names)
df['target'] = cancer['target']     # target -> 0 : 악성종양, 1 : 양성종양
print(df.head())

pos = df.loc[df['target']==1]
neg = df.loc[df['target']==0]

## 실습을 위해 양성환자 357개, 악성환자 5개가 되도록 설정해준다.
sample = pd.concat([pos, neg[:5]], sort=True)

x_train, x_test, y_train, y_test = train_test_split(sample.drop('target', 1), sample['target'], random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

print("유방암 환자 예측 : ", (pred==y_test).mean())

## 여기서 우스운 가정
## 돌팔이 의사가 자신의 진단률이 높다는 것을 뻥치기 위해 무조건 양성이라고 우긴다음 확률을 나타냄
my_prediction = np.ones(shape=y_test.shape)
print("가짜 유방암 환자 예측 : ", (my_prediction==y_test).mean())   # 실제 데이터 예측보다 확률이 높다.
## 정확도(accuracy)만 보고 분류기의 성능을 판별하는 것은 위와 같은 오류에 빠질 수 있다.
## 이를 보완하려 생격난 지표들이 있다.

# 오차 행렬(confusion matrix)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred))

sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Reds')
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()


# 정밀도 (precision)
from sklearn.metrics import precision_score, recall_score

## 양성 예측 정확도 : TP / (TP + FP)
precision_score(y_test, pred)     # 무조건 양성으로 판단했을 경우 좋은 정밀도를 얻기 때문에 유용하지는 않다.
print("양성 예측 정확도 : ",precision_score(y_test, pred))

## 재현율 (recall) : TP / (TP + FN)
recall_score(y_test, pred)      # 정확하게 감지한 양성 샘플의 비율 -> 민감도(sensitivity) 혹은 True Positive Rate(TPR)이라고도 불린다.
print("재현율 정확도 : ", recall_score(y_test, pred))


# f1 score
## 정밀도와 재현율의 조화 평균을 나타내는 지표
## https://miro.medium.com/max/918/1*jCu9fNZSOhSRHVJ2cBTegg.png

from sklearn.metrics import f1_score
f1_score(y_test, pred)
print("f1 score 정확도 : ", f1_score(y_test, pred))


