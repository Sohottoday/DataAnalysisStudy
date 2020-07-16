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

