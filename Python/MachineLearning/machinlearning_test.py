import matplotlib.pylab as plt
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


# scipy에서 scikit-learn 알고리즘을 구현할 때 가장 중요한 기능은 scipy.sparse 모듈
# 이때 희소 행렬기능은 주요 기능 중의 하나이다.
# 희소 행렬(sparse matrix) : 0을 많이 포함한 2차원 배열

from scipy import sparse

b1 = np.eye(4, dtype=int)

print("Numpy 배열 : \n{}".format(b1))

# sparse.csr_matrix() : 0이 아닌 원소만 저장
# CSR(Compressed Sparse Row) : 행의 인덱스를 압축해서 저장

sparse_matrix = sparse.csr_matrix(b1)
print("Scipy의 CSR 행렬 : \n{}".format(sparse_matrix))

b2 = np.eye(5, k=-1, dtype=int)
print(b2)

sparse_matrix = sparse.csr_matrix(b2)
print("Scipy의 CSR 행렬2 : \n{}".format(sparse_matrix))

b3 = np.arange(16).reshape(4, 4)
print(b3)

x = np.diag(b3)
print(x)

y = np.diag(np.diag(b3))
print(y)

sparse_matrix = sparse.csr_matrix(y)
print("SciPy의 CSR 행렬3 : \n{}".format(sparse_matrix))

# 희소행렬을 직접 만들 때 사용하는 format
## COO 포맷(Coordinate 포맷), 메모리 사용량을 많이 줄여준다.

data = np.ones(4)
print(data)

row_indices = np.arange(4)
col_indices = np.arange(4)

eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현 : \n{}".format(eye_coo))
# 이러한 예제로는 데이터 양이 적기 때문에 체감하지 못하지만 빅데이터를 다룰 때 메모리양이 부족해서 오류가 나는 경우를 방지해 준다.


# 내장 데이터셋 불러오기
from sklearn.datasets import load_iris

irisData = load_iris()      # Bunch 클래스 객체라고 한다. Python의 딕셔너리 객체 형태와 유사하다.
print(irisData.keys())

print(irisData['target_names'])

print(irisData['data'].shape)

# 데이터를 훈련용 data와 테스트용 data로 나누는 과정이 필요하다.
# 보통 7:3 혹은 8:2 비율로 나눠준다.
# train_test_split 로 수행한다.
# train_test_split 모듈은 sklearn.model_selection에 존재한다.
# train_test_split 수행하기 전 데이터의 섞는 과정이 필요하다.
# scikit-learn 에서 데이터는 보통 대문자 X로 표기하고 label은 소문자 y로 표현한다.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(irisData['data'], irisData['target'], random_state=0)

# train_test_split()의 리턴 타입은 모두 numpy 배열이다.
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)


# KNN : K-Nearest Neighbors => K-최근접 이웃 알고리즘
## 사용하기 쉬운 분류 알고리즘(분류기) 중의 하나이다.
## 새로운 데이터를 훈련 데이터 중 가장 가까운 데이터를 찾아내는 것
## k의 의미는 가장 가까운 이웃 하나를 의미하는 것이 아니라 훈련데이터에서 새로운 데이터에 가장 가까운 k개의 이웃을 찾는다는 의미

# KNN을 사용하기 위해서는 neighbors 모듈의 KNeighborsClassifier 함수 사용
# KNeighborsClassifier() 함수의 중요한 매개변수는 n_neighbors
# 이 매개변수는 이웃의 개수를 지정하는 매개변수

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터셋을 가지고 모델을 만들려면 fit 메서드를 사용한다
# fit 메서드의 리턴값은 knn객체를 리턴한다.

knn.fit(X_train, y_train)
print(knn.fit(X_train, y_train))

# 채집한 붓꽃의 새로운 데이터(샘플)라고 가정하고 Numpy 배열로 특성값을 만든다.
# scikit-learn에서는 항상 데이터가 2차원 배열일 것으로 예측해야 한다.

X_newData = np.array([[5.1, 2.9, 1, 0.3]])

# knn 객체의 predict() 메서드를 사용하여 예측할 수 있다.
prediction = knn.predict(X_newData)
print('예측 : ', prediction)

print('예측한 품종의 이름 : ', irisData['target_names'][prediction])

y_predict = knn.predict(X_test)
print(y_predict)

x = np.array([1, 2, 3, 2])

# 정확도를 계산하기 위해 numpy의 mean() 메서드 사용
# knn객체의 score() 메서드를 사용해도 된다.
print(np.mean(y_predict == y_test))     # 1에 가까울수록 정확도가 높다는 의미 
## 이 의미는 y_predict가 X_test에 대한 예측값과의 비교
print(knn.score(X_test, y_test))


# 머신 러닝의 용어 정리

## iris 분류 문제에 있어 각 품종을 클래스라고 한다.
## 개별 붓꽃의 품종은 레이블이라고 한다.
## 붓꽃의 데이터셋은 두 개의 Numpy 배열로 이루어져 있다.
### 하나는 데이터, 다른 하나는 출력을 가지고 있다.
## scikit-learn에서는 데이터는 X로 표기하고, 출력은 소문자 y로 표기한다.
## 이 때 배열 X는 2차원 배열이고 각 행은 데이터포인트(샘플)에 해당한다.
## 각 컬럼(열)은 특성이라고 한다.
## 배열 y는 1차원 배열이고, 각 샘플의 클래스 레이블에 해당한다.

from sklearn import svm, metrics
import random, re

csv = []

with open('iris.csv', 'r', encoding='utf-8') as fp:
    # 한 줄씩 읽어오기 
    for line in fp:
        line = line.strip()     # 줄바꿈 제거
        cols = line.split(',')      # 컴마 기준으로 컬럼을 잘라내겠다는 의미
        # 문자열 데이터를 숫자로 변환하기
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols))
        csv.append(cols)

# 헤더 제거(컬럼명 제거)
del csv[0]

# 데이터를 섞어주기
random.shuffle(csv)

# 훈련(학습) 데이터와 테스트 데이터로 분리하기
total_len = len(csv)
train_len = int(total_len * 2/3)

train_data = []
train_label = []

for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
# 학습
clf.fit(train_data, train_label)

# 테스트
predict_label = clf.predict(test_data)

# 정확도 구하기
ac_score = metrics.accuracy_score(test_label, predict_label)

print('정확도 : ', ac_score)



################
# pandas 이용하기
csv = pd.read_csv('iris.csv')

# 데이터와 레이블 분리하기
csv_data = csv[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
csv_label = csv['Name']

# 훈련 데이터와 테스트 데이터로 분리하기
X_train, X_test, y_train, y_test = train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)
print('정확도 : ', ac_score)


####################
# 2만명 데이터 만들기(csv 파일)
# Body Mass Index(체질량 지수:bmi) 데이터 만들어보기

def bmi_func(height, weight):
    bmi = weight/(height/100) ** 2
    if bmi < 18.5 : return "저체중"
    if bmi < 25 : return "정상 체중"
    return "비만"

fp = open('bmi.csv', 'w', encoding='utf-8')
fp.write('height, weight, label\r\n')

# 데이터 생성하기
cnt = {'저체중' : 0, '정상': 0, '비만' : 0}

for i in range(10000):
    h = random.randint(120, 200)
    w = random.randint(35, 90)
    label = bmi_func(h, w)
    cnt[label] += 1
    fp.write('{0}, {1}, {2}\r\n'.format(h, w, label))
fp.close()
print('ok', cnt)


# 퍼셉트론 식

# 가중치와 편향을 도입한 퍼셉트론 식
#       0     w1 * x1 + w2 * x2 <= seta
# y
#       1     w1 * x1 + w2 * x2 > seta
# x1, x2는 입력신호, y는 출력신호, w1, w2는 가중치를 의미
# (w는 weight를 의미) seta는 임계값

# w1 * x1 + w2 * x2 값이 임계값 이하일 때는 0을 출력, 임계값보다 클때 1을 출력
# 가중치가 클수록 해당신호가 강해진다 

# AND 게이트 진리표
# x1        x2          y
#------------------------
# 0         0           0
# 0         1           0
# 1         0           0
# 1         1           1

# 입력이 모두 1일때만 1을 출력한다.
# 퍼셉트론으로 AND게이트 표현하고자 할 때는 w1, w2, 세타 값을 어떤 값으로 설정할 것인지 생각해야 한다.

# AND 게이트를 만족하는 w1, w2, seta의 조합은 무수히 많다.      // 0.5가 가중치이다.
# ex) (w1, w2, seta) -> (0.5, 0.5, 0.8)   0.8보다 크면 1
# 1 * 0.5 + 1 * 0.5 는 0.8보다 크기 때문에 1


# NAND(Not AND) 게이트 진리표
# AND 게이트와 반대

# NAND 게이트를 표현하기 위한 매개변수의 값
# (w1, w2, seta) -> (-0.5, -0.5, -0.7)

# OR게이트 진리표
# x1        x2          y
#------------------------
# 0         0           0
# 0         1           1
# 1         0           1
# 1         1           1

# 입력신호 중 하나 이상이 1이면 출력이 1이되는 논리 회로

# 퍼셉트론 표현할 때는 가중치와 임계값을 설정하여 표현할 수 있다.
# 이 매개변수(w, seta)의 값을 적절히 조절하면 AND, NAND, OR 게이트를 모두 표현 가능하다.

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.8
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

# 위의 퍼셉트론 식에서 theta를 -b로 치환하면
# w1 * x1 + w2 * x2 <= -b       ==      b + w1 * x1 + w2 * x2 <=0
# w1 * x1 + w2 * x2 > -b        ==      b + w1 * x1 + w2 * x2 > 0
# 이때 b(bias)를 편향이라고 한다.
# 퍼셉트론은 입력신호에 가중치를 곱한 값과 편향을 합하여, 그 값이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력한다.

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print(w * x)
print(np.sum(w * x) + b)

def AND1(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7            # 여기서 0.7은 무슨 의미인가?
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1



# XOR 게이트 진리표

# x1        x2          y
#------------------------
# 0         0           0
# 0         1           1
# 1         0           1
# 1         1           0

# XOR 게이트는 배타적 논리합이라고 한다.
# x1과 x2 중 어느 한쪽이 1일 때만 1을 출력하는 논리회로

# 퍼셉트론으로는 XOR 게이트를 구현할 수 없다.
# 직선 하나로는 XOR 게이트 출력을 구분할 수 없다.
# 퍼셉트론(단층 퍼셉트론)은 직선 하나로 나눈 영역만 표현할 수 있다는 한계가 있다.
# 선형 : 직선의 영역을 선형 영역
# 비선형 : 곡선의 영역


# 다층 퍼셉트론(Multi Layer Perceptron)

# 단층 퍼셉트론으로는 XOR 게이트를 표현할 수 없다.
# 즉, 단층 퍼셉트론으로는 비선형 영역을 분리할 수 없다.


# 기존 게이트(AND, OR, NAND) 조합하여 층을 쌓으면 XOR게이트를 구현할 수 있게 된다.

# 게이트를 조합한 XOR 게이트의 진리표

# x1      x2      s1      s2      y
#----------------------------------------
#  0       0       1       0      0
#  0       1       1       1      1
#  1       0       1       1      1
#  1       1       0       1      0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])      # 가중치
    b = 0.7     # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else :
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
    
print("--------XOR--------")
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))



# 신경망 : 입력층과 은닉층 출력층으로 이루어져 있다 

# 은닉층(사람 눈에 보이지 않는 층)은 중간층을 말한다.

# 입력층(0층), 은닉층(1층), 출력층(3층)은 퍼셉트론의 신호전달 방식과 공통점이 있다.

# 활성화 함수를 이용한 퍼셉트론 식
# y = h(b + w1 * x1 + w2 * x2)
#           0 (x <= 0)  이때 x는 입력신호의 총 합
# h(x) = 
#           1 (x > 0)

# 입력신호의 총합을 출력신호로 변환하는 함수를 "활성화 함수"라고 한다.
# a = b + w1 * x1 + w2 * x2     (입력신호의 총합)
# y = h(a)

# 위의 식은 가중치가 있는 입력신호와 편향을 계산하고 이를 a라고 하면 a를 함수 h()에 넣어 y를 출력하는 흐름을 보여준다.

# 시그모이드(sigmoid) 함수
# 신경망에서 자주 이용하는 활성화 함수 중의 하나
#                 1
#   h(x) = -----------------
#            1 + exp(-x)

# 계단 함수
## 임계값을 경계로 출력이 바뀌는 함수

def step_func(x):
    return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_func(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)         
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# 신경망의 내적(가중치만 적용)

X = np.array([[1, 2], [3, 4]])          # 입력신호
W = np.array([[1, 3, 5], [2, 4, 6]])        # 1, 3, 5는 x1의 가중치 / 2, 4, 6은 x2의 가중치

Y = np.dot(X, W)


# 신경망 파이썬 구현(sigmoid 함수 사용)

print(Y)

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
print(A1)

z1 = sigmoid(A1)
print(z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(z1, W2) + B2
print(A2)

Z2 = sigmoid(A2)
print(Z2)


def identity_function(x):           # 항등 함수(시그마 함수)
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3)           # Y = A3
print(Y)
print(A3)


# 다층 신경망

## 가중치와 편향을 초기화해주는 함수

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

## 입력 신호를 출력으로 변환하는 처리과정(순방향)

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])

y = forward(network, x)
print(y)


# 기계 학습의 문제
## 분류(classfication)
## 회귀(regression)

## 신경망에서는 회귀에 사용하는 활성함수로 항등함수를 사용하며, 분류에 사용하는 활성함수로 소프트맥스 함수를 사용한다.
## 다중 분류를 할 때 소프트맥스 함수를 사용한다.(기본 분류는 시그모이드 함수 사용)

## 소프트맥스 함수 정의
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.2, 2.9, 4.0])

print(softmax(a))

## 위 같은 경우에는 지나치게 큰 값이 나와 overflow 에러가 뜰 수 있다

## 개선된 소프트맥스 함수 정의

def upgrade_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)       # 오버플로우를 막기 위해
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

print(upgrade_softmax(a))           # 값이 항상 0에서 1 사이의 값이 출력된다.
print(np.sum(upgrade_softmax(a)))       # 값을 모두 더하면 1이 된다.
# 즉, 이 출력값은 확률로 해석할 수 있다. 문제를 통계적으로 대응할 수 있다.

# 각 원소의 대소 관계는 소프트맥스 함수의 출력값의 대소관계와 동일하다
# 신경망을 이용한 분류에서는 출력층의 소프트맥스 함수를 생략해도 무방하다.
# 소프트맥스 함수를 적용하지 않더라도 결과는 동일하기 때문
# 현업에서는 실질적으로 지수 함수 계산에 드는 자원 낭비를 줄일 수 있기 때문에 소프트맥스 함수를 생략한다.

# TensorFlow
## 구글이 오픈소스로 공개한 머신러닝 라이브러리
## 딥러닝을 비롯한 여러 머신러닝에 사용되는 라이브러리
## 대규모 숫자 계산을 해주는 라이브러리
## 일반적인 프로그래밍 방식과는 약간 다른 개념들을 포함한다.

## import tensorflow as tf

# tensorflow의 자료형
## tensor : 다양한 수학식을 계산하기 위한 가장 기본적이고 중요한 자료형
## Rank와 Shape라는 개념

## Rank값
## 0 : 스칼라
## 1 : 벡터
## 2 : 행렬
## 3 이상의 n : n-Tensor 또는 n차원 텐서

## Shape : 각 차원 요소 개수를 의미, 텐서의 구조를 의미

## dtype : 해당 텐서에 담긴 요소들의 자료형, string, float, int 등

import tensorflow as tf

hello = tf.constant('hello tensorflow')
print(hello)

# tensorflow 프로그램 구조
## 그래프 생성
## 그래프 실행

## 그래프 : tensor들의 연산 모음
## 지연실행(lazy evaluation) : 함수형 프로그래밍에서 많이 사용
## 텐서와 텐서의 연산들을 미리 정의하여 그래프를 만들고, 필요할 때 연산을 실행하는 코드를 넣어 원하는 시점에 실제 연산을 수행하도록 하는 방식

# tensorflow 프로그래밍의 장점
## 모델 구성과 실행을 분리하여 프로그램을 깔끔하게 작성할 수 있다.

# tensorflow 프로그래밍 시 반드시 알아야 할 개념 두가지
## placeholder : 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수와 같은 개념
## 변수 : 그래프를 최적화하는 용도로 텐서플로우가 학습한 결과를 갱신하기 위해 사용하는 것
##       (변수의 값들이 신경망의 성능을 좌우한다.)






# 시그모이드 함수는 단순하게 생각하면 S형태의 그래프라고 생각하면 된다.
# 신경망에서는 입력값에 따라 중요도가 다르다.(중요할 경우 큰 값)
# 값이 아무리 작아도 최소값은 0이고 아무리 값이 커도 최대값은 1(계단함수와 시그모이드 함수의 공통점)

# 계단함수와 시그모이드 함수의 공통점
## 비선형 함수(직선 하나로는 그릴수 없는 함수) - 따라서, 신경망에서는 활성화함수로 비선형함수를 사용해야 한다.
## (선형함수를 사용할 경우 신경망 층을 깊게할수가 없다.)

# ReLU 함수
## Rectified Linear Unit 함수 : 입력이 0을 넘으면 그 입력을 그대로 출력하고 0 이하이면 0을 출력하는 함수
#         x (x > 0)     - 0보다 크면 x 그대로 출력
# h(x) = 
#         0 (x < 0)     - 0보다 작으면 0 출력

def relu(x):
    return np.maximum(0, x)     # maximum : 두 입력중에 큰 값을 반환하는 함수


# 행렬의 곱(행렬의 내적)
## 2차원 배열의 내적(행렬의 곱)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))        # 내적을 구하는 함수

# 행렬의 내적은 대응하는 차원의 원소 수를 일치시켜야 한다.
