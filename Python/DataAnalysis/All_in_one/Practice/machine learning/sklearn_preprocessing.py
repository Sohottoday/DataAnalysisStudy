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


# Label Encoding : 문자(categorical)를 수치(numerical)로 변환
## 학습을 위해서 모든 문자로된 데이터는 수치로 변환하여야 한다.
def convert(data):
    if data == 'male':
        return 1
    elif data == 'female':
        return 0

train['Sex'].apply(convert)

## 위의 과정 필요 없이 sklearn의 LabelEncoder를 활용해 쉽게 변환할 수 있다.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Sex_num'] = le.fit_transform(train['Sex'])      # 라벨 인코딩을 할 column을 학습시키고 변환시킨다. 학습시킨 데이터는 'Sex_num'이라는 새 항목에 담았음
print(train['Sex_num'].value_counts())      ## 남 녀 숫자데이터로 변환된 것을 확인할 수 있다.
print(le.classes_)      ## 변환된 클래스들을 확인해보기
print(le.inverse_transform([0, 1, 1, 0, 0, 1, 1]))           ## 다시 원래의 데이터로 돌리고 싶을때에는 inverse_transform을 활용할 수 있다.

## NaN 값이 포함되어 있다면, LabelEncorder가 제대로 작동하지 않는다.
## 이러한 경우 결측치 처리를 먼저 해준 뒤 작업한다.
train['Embarked'] = train['Embarked'].fillna('S')
le.fit_transform(train['Embarked'])


# 원 핫 인코딩(One Hot Encoding)
print("---------one hot encoding-----------")
train = pd.read_csv('train.csv')

## Embarked를 원 핫 인코딩 해보려 한다.
print(train['Embarked'].value_counts())
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked_num'] = LabelEncoder().fit_transform(train['Embarked'])
print(train['Embarked_num'].value_counts())

## Embarked는 탑승 항구의 이니셜을 나타낸다.
## 그런데, 우리는 LabelEncoder를 통해 수치형으로 변환해주었다.
## 하지만, 이대로 데이터를 기계학습 시키면 기계는 데이터 안에서 관계를 학습한다.
## 즉 'S'=2, 'Q'=1 이라고 되어 있는데 Q+Q=S 가 된다 라고 학습한다.
## 그렇기 때문에, 독립적인 데이터는 별도의 column으로 분리하고, 각각의 컬럼에 해당 값에만 True 나머지는 False를 갖는다. 이것을 '원 핫 인코딩 한다' 라고 한다.

one_hot = pd.get_dummies(train['Embarked_num'])
one_hot.columns = ['C', 'Q', 'S']
print(one_hot)
## 위와 같이 column을 분리시켜 카테고리형->수치형으로 변환하면서 생기는 수치형 값의 관계를 끊어주면서 독립적인 형태로 바꾸어 준다.
## 원핫인코딩은 카테고리(계절, 항구, 성별, 종류, 한식/일식/중식...)의 특성을 가지는 column에 대해서 적용한다.

