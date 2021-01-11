import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

"""
전처리(pre-processing)
1. 결측치 - Imputer
2. 이상치
3. 정규화(Normalization)
4. 표준화(Standardization)
5. 샘플링(over/under sampling)
6. 피처 공학(Feature Engineering)
    - feature 생성/연산
    - 구간 생성, 스케일 변형 ..

# 정규화(Normalization)
    - 0 ~ 1 사이의 분포로 조정

# 표준화
    - 평균을 0, 표준편차를 1로 맞춤
"""

train = pd.read_csv('train.csv')

"""
PassengerId : 승객 아이디
Survived : 생존여부 = 1:생존, 0:사망
Pclass : 등급
Name : 성함
Sex : 성별
Age : 나이
Sibsp : 형제, 자매, 배우자 수
Parch : 부모, 자식 수
Ticket : 티켓 번호
Fare : 요금
Cabin : 좌석 번호
Embarked : 탑승 항구
"""

# 전처리 : train / validation 세트 나누기
## 먼저 feature 와 label을 정의한다
## feature / label을 정의했으면 적절한 비율로 train / validation set을 나눈다.

feature = [
    'Pclass', 'Sex', 'Age', 'Fare'
]

label = [
    'Survived'
]

from sklearn.model_selection import train_test_split
## test_size : validation set 에 할당할 비율(20% -> 0.2)
## shuffle : 셔플 옵션(기본 True)
## random_state : 랜덤 시드값
## return 받는 데이터의 순서가 중요하다

x_train, x_valid, y_train, y_valid = train_test_split(train[feature], train[label], test_size=0.2, shuffle=True, random_state=30)
print("훈련 값 : ",x_train.shape, y_train.shape)
print("예측하려는 값 : ", x_valid.shape, y_valid.shape)


# 전처리 : 결측치
## 결측치를 확인하는 방법은 pandas의 isnull()
## 그리고 합계를 구하는 sum()을 통해 한 눈에 확인할 수 있다.

print(train.info())
print(train.isnull().sum())

## 개별 column의 결측치에 대하여 확인하는 방법
print("개별 결측치 : ",train['Age'].isnull().sum())

## 1. 수치형(Numerical Column)데이터에 대한 결측치 처리
# train['Age'].fillna(0).describe()
print(train['Age'].fillna(train['Age'].mean()).describe())

## imputer : 2개 이상의 column을 한 번에 처리할 때
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')            # strategy='mean'은 빈 값을 평균으로 채워주겠다는 의미
### fit()을 통해 결측치에 대한 학습을 진행
imputer.fit(train[['Age', 'Pclass']])       # age와 pclass 컬럼을 fit()을 통해 결측치에 대한 학습을 진행한다.
### transform() 은 실제 결측치에 대한 처리를 해주는 함수
result = imputer.transform(train[['Age', 'Pclass']])
print(result)
train[['Age', 'Pclass']] = result
print(train[['Age', 'Pclass']].isnull().sum())

### 위의 방법은 과정이 길어 번거로운 느낌이 든다.
### fit_transform()은 fit()과 transform()을 한 번에 해주는 함수
train = pd.read_csv('train.csv')

imputer = SimpleImputer(strategy='median')
result = imputer.fit_transform(train[['Age', 'Pclass']])

## (Categorical Column) 데이터에 대한 결측치 처리
train['Embarked'].fillna('S')

### Imputer를 사용하는 경우 : 2개 이상의 column을 처리할 때
imputer = SimpleImputer(strategy='most_frequent')           # most_frequent : 가장 빈도가 많은것으로 채운다는 의미
result = imputer.fit_transform(train[['Embarked', 'Cabin']])



