# 회귀(regression) 예측
"""
수치형 값을 예측(Y의 값이 연속된 수치로 표현)
ex)
    주택 가격 예측
    매출액 예측
"""

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)          # 숫자를 표시할 때 e-5 이런식으로 표기하는것을 0.00001 과 같이 바꿔줌

from sklearn.datasets import load_boston
"""
# 보스턴 집값 데이터
## 컬럼 소개
- CRIM : 범죄율
- ZN : 25,000 평방 피트 당 주거용 토지의 비율
- INDUS : 비소매(non-retail)비즈니스 면적 비율
- CHAS : 찰스 강 더미 변수(통로가 하천을 향하면 1; 그렇지 않으면 0)
- NOX : 산화 질소 농도(천만분의 1)
- RM : 주거 당 평균 객실 수
- AGE : 1940년 이전에 건축된 자가 소유 점유 비율
- DIS : 5개의 보스턴 고용 센터까지의 가중 거리
- RAD : 고속도로 접근성 지수
- TAX : 10,000달러 당 전체 가치 재산세율
- PTRATIO : 도시 별 학생-교사 비율
- B : 1000(Bk-0.63)^2  여기서 Bk는 도시 별 검정 비율
- LSTAT : 인구의 낮은 지위
- MEDV : 자가 주택의 중앙값(1,000 달러 단위)
"""

data = load_boston()

df = pd.DataFrame(data['data'], columns=data['feature_names'])

# Y 데이터인 price도 데이터 프레임에 추가
df['MEDV'] = data['target']

print(df.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop('MEDV', 1), df['MEDV'])
print(x_train.shape, x_test.shape)


# 평가 지표 만들기
## MSE(Mean Squared Error)(평균제곱오차) : 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 값(오차율 구하는 방법)
## MAE(Mean Absolute Error) : 예측값과 실제값의 차이에 대한 절대값에 대하여 평균을 낸 값
## RMSE(Root Mean Squared Error) : 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 뒤 루트를 씌운 값
## 위의 평가지표 만들어보기
pred = np.array([3, 4, 5])      # 임시 예측 값
actual = np.array([1, 2, 3])    # 임시 실제 값

def my_mse(pred, actual):
    return ((pred - actual)**2).mean()

def my_mae(pred, actual):
    return np.abs(pred - actual).mean()

def my_rmse(pred, actual):
    return np.sqrt(my_mse(pred, actual))

print("MSE : ", my_mse(pred, actual))
print("MAE : ", my_mae(pred, actual))
print("RMSE : ", my_rmse(pred, actual))


# sklearn의 평가지표 활용하기
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("mean_absolute_errer : ", mean_absolute_error(pred, actual))
print("mean_squared_error : ", mean_squared_error(pred, actual))

# 모델별 성능을 위한 함수(참고만 할 것)
import matplotlib.pyplot as plt
import seaborn as sns

my_predictions = {}

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive', 
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray', 
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]

def plot_predictions(name_, pred, actual):
    df = pd.DataFrame({'prediction': pred, 'actual': y_test})
    df = df.sort_values(by='actual').reset_index(drop=True)

    plt.figure(figsize=(12, 9))
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()

def mse_eval(name_, pred, actual):
    global predictions
    global colors

    plot_predictions(name_, pred, actual)

    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    
    length = len(df)
    
    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mse'])
    
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')
        
    plt.title('MSE Error', fontsize=18)
    plt.xlim(min_, max_)
    
    plt.show()

def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True


# LinearRegression(선형 회귀)
from sklearn.linear_model import LinearRegression

model = LinearRegression(n_jobs=-1)
model.fit(x_train, y_train)
pred = model.predict(x_test)
mse_eval('LinearRegression', pred, y_test)

