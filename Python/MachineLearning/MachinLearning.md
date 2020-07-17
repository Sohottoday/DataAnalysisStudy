

# 머신러닝

- 명시적으로 프로그래밍을 하지 않고도 컴퓨터가 학습할 수 있는 능력을 갖게 하는 것
- 사용 예
  - 데이터 마이닝
    - 클릭 기록, 의료기록, 유전자 분석 등
  - 수작업으로 프로그래밍 할 수 없는 것들
    - 자율 운행 헬리콥터, 얼굴 인식, 스팸 필터 등
  - 개개인의 유저에게 최적화된 추천 알고리즘
    - 상품 추천, 영화 추천

### 머신러닝 알고리즘 분류

- 학습 방법

  - 지도학습(Supervised Learning)
    - 학습 데이터마다 레이블을 가지고 있음 => 입력이 있다면 출력값이 있는 경우를 의미
  - 비지도학습(Unsupervised Learning)
    - 학습 데이터가 레이블을 가지고 있지 않음
      - ex) Clustering
  - 준지도학습(Semi_Supervised Learning)
    - 학습 데이터가 약간의 레이블을 가지고 있음
  - 강화학습(Reinforcecment Learning)
    - 최종 출력이 바로 주어지지 않고 시간이 지나서 주어지는 경우

  

  ### 머신러닝의 구성요소

  #### 데이터 준비

  - 훈련(학습) 데이터		=> 모델을 만들기 위해 학습하기 위한 데이터
  - 검증 데이터
  - 테스트 데이터




#### 모델 표현 방법

- 의사결정 트리
  - 귀납적 추론방법, 기호주의, 철학과 심리학, 논리학 기반

- 신경망 기반
  - 연결주의, 두뇌를 분석하고 모방하여 신경과학과 물리학 기반
- KNN, 서포트벡터머신
  - 유추주의, 유사성을 근거로 추정하면서 학습
- 베이지안 모델
  - 학습이 확률 추론의 한 형태로 믿으며 통계학 기반
- 유전 알고리즘
  - 진화주의,  유전학과 진화생물학 기반
- 모델 앙상블



#### 모델 평가 방법

- 에러의 제곱
- 정확도
- 우도(가능도)
- 정밀도와 재현율
- 엔트로피



#### scikit-learn

- 파이썬 머신러닝 라이브러리

- 머신 러닝의 응용분야
  - 분류(Classfication) : 특정 데이터에 레이블을 붙여 분류할 수 있다.
  - 클러스터링(Clustring) : 값의 유사성을 기반으로 데이터를 여러 그룹으로 나누는 것
  - 추천(Recommendation) : 과거의 데이터를 기반으로 미래의 데이터를 예측
  - 회귀(Regression) : 과거의 데이터를 기반으로 미래의 데이터를 예측하는 것
  - 차원축소 : 데이터의 특성을 유지하면서 데이터의 양을 줄여주는 것
- `from sklearn import svm`
- 활용 코드

``` python
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
# 예측결과 :  [0 1 1 0]

ok = 0; total = 0

for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok +=1
    total += 1
print('정확도 : ', ok, '/', total, '=', ok/total)
# 정확도 :  4 / 4 = 1.0

## ok 값과 total 값이 같다면 정확도가 100% 라는 의미

```

---

``` python
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
clf.fit(trainingData, label)		# 학습시키기
pre = clf.predict(trainingData)		# 예측하기

# 정확도 측정(정답률 확인)
# metrics 모듈의 accuracy_score()
## import sklearn import metrics 필요

accuracy = metrics.accuracy_score(label, pre)
print("정확도 : ", accuracy)
# 정확도 :  1.0
```



- scipy의 sparse 모듈
  - 희소 행렬을 구하는 모듈
- numpy에서 특수행렬을 만드는 함수
  - eye(N, M=, k=, dtype= ) : 항등행렬
    - M : 열의 수
    - k : 대각의 위치

``` python
print(np.eye(4, M=3, k=1, dtype=int))
# [[0 1 0]
#  [0 0 1]
#  [0 0 0]
#  [0 0 0]]
```

- diag() 함수는 정방행렬에서 대각 요소만 추출하여 벡터를 만든다.
  - diag(v, k=, )
    - k : 시작 위치

``` python
x = np.eye(5, dtype=int)
print(np.diag(x))
# [1 1 1 1 1]

x = np.arange(9).reshape(3, 3)
print(x)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(np.diag(x))
# [0 4 8]

print(np.diag(np.diag(x)))			# diag()함수는 반대로 벡터 요소를 대각 요소로 하는 정방 행렬을 만들 수 있다.
# [[0 0 0]
#  [0 4 0]
#  [0 0 8]]
```

