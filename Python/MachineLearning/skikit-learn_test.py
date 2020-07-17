# scikit-learn : 파이썬 머신러닝 라이브러리

# 머신러닝의 응용분야
## 분류(Classfication) : 특정 데이터에 레이블을 붙여 분류할 수 있다.
## 클러스터링(Clustring) : 값의 유사성을 기반으로 데이터를 여러 그룹으로 나누는 것
## 추천(Recommendation) : 특정 데이터를 기반으로 다른 데이터를 추천하는 것
## 회기(Regression) ; 과거의 데이터를 기반으로 미래의 데이터를 예측하는 것
## 차원축소 : 데이터의 특성을 유지하면서 데이터의 양을 줄여주는 것

from sklearn import svm

# XOR 연산 활용
xor_data = [
    [0, 0, 0],      # 0과 0이면 0이 나온다는 의미
    [0, 1, 1],      # 0과 1이면 1이 나온다는 의미
    [1, 0, 1],      # 1과 0이면 1이 나온다는 의미
    [1, 1, 0]       # 1과 1이면 0이 나온다는 의미
]

# 주어진 데이터를 분리한다. (학습 데이터와 레이블을 분리)
training_data = []
label = []

for row in xor_data:
    p = row[0]
    q = row[1]
    result = row[2]

    training_data.append([p, q])
    label.append(result)

# SVM 알고리즘을 사용하는 머신러닝 객체 생성
## SVM : 분류, 회귀 알고리즘
### SVC : 분류에 해당하는 알고리즘
### SVR : 회귀에 해당하는 알고리즘
clf = svm.SVC()

# fit() 메서드 : 학습기계에 데이터를 학습시킨다.
clf.fit(training_data, label)

# predic() 메서드 : 학습 데이터를 이용하여 예측한다.
pre = clf.predict(training_data)
print('예측결과 : ',pre)

ok = 0; total = 0

for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok +=1
    total += 1
print('정확도 : ', ok, '/', total, '=', ok/total)
## ok 값과 total 값이 같다면 정확도가 100% 라는 의미

import pandas as pd
import numpy as np
from sklearn import metrics     # support vector machine

# XOR 연산 데이터
inputData = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0]
]

xor_df = pd.DataFrame(inputData)

# 학습데이터와 레이블을 분리
trainingData = xor_df.loc[:, 0:1]
label = xor_df.loc[:, 2]

clf = svm.SVC()
clf.fit(trainingData, label)     # 학습시키기
pre = clf.predict(trainingData)         # 예측하기

# 정확도 측정(정답률 확인)
# metrics 모듈에 accuracy_score(), import sklearn import metrics 필요

accuracy = metrics.accuracy_score(label, pre)
print("정확도 : ", accuracy)

# scipy의 sparse 모듈 => 희소 행렬을 구하는 모듈

# numpy에서 특수행렬을 만드는 함수
# eye(N, M=, k=, dtype=) : 항등행렬
## M : 열의 수
## k : 대각의 위치
print(np.eye(4, M=3, k=1, dtype=int))

# diag() 함수는 정방행렬에서 대각 요소만 추출하여 벡터를 만든다.
# diag(v, k=, )
## k : 시작 위치
x = np.eye(5, dtype=int)
print(np.diag(x))

x = np.arange(9).reshape(3, 3)
print(x)
print(np.diag(x))
print(np.diag(np.diag(x)))      # diag()함수는 반대로 벡터 요소를 대각 요소로 하는 정방 행렬을 만들 수 있다.