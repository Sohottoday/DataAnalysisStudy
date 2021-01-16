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


# 규제(Regularization)
## 학습이 과대적합 되는 것을 방지하고자 일종의 패널티(penalty를 부여하는 것)
"""
L2 규제(L2 Regularization)
    각 가중치 제곱의 합에 규제 강도(Regularization Strength) λ를 곱한다.
    λ를 크게 하면 가중치가 더 많이 감소되고(규제를 중요시함), λ를 작게 하면 가중치가 증가한다(규제를 중요시하지 않음).
L1 규제(L1 Regularization)
    가중치의 제곱의 합이 아닌 가중치의 합을 더한 값에 규제 강도(Regularization Strength) λ를 곱하여 오차에 더한다.
    어떤 가중치(w)는 실제로 0이 된다. 즉, 모델에서 완전히 제외되는 특성이 생기는 것이다.
L2 규제가 L1규제에 비해 더 안정적이라 일반적으로는 L2 규제가 더 많이 사용된다.

릿지(Ridge) - L2 규제
Error = MSE + aw^2

라쏘(Lasso) - L1 규제
Error = MSE + a|w|
"""

## Ridge(릿지)
from sklearn.linear_model import Ridge

# 값이 커질수록 큰 규제
alphas = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]     # 규제 강도 설정

for alpha in alphas:            # 각 규제 강도별로 모델 성능이 어떻게 나오는지
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    mse_eval('Ridge(alpha={})'.format(alpha), pred, y_test)         # 이번 값은 규제가 낮을수록 성능이 좋아졌지만 가장 성능이 좋은값은 의외로 100

print(ridge.coef_)      # 각각의 가중치들을 볼 수 있다.

## alpha값에 따른 coef의 차이를 확인해보기 위한 함수 생성
def plot_coef(columns, coef):
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns=['feature', 'coef']
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(len(coef_df)), coef_df['coef'])
    idx = np.arange(len(coef_df))
    ax.set_yticks(idx)
    ax.set_yticklabels(coef_df['feature'])
    fig.tight_layout()
    plt.show()

plot_coef(x_train.columns, ridge.coef_)

ridge_100 = Ridge(alpha=100)
ridge_100.fit(x_train, y_train)
ridge_pred_100 = ridge_100.predict(x_test)

ridge_001 = Ridge(alpha=0.001)
ridge_001.fit(x_train, y_train)
ridge_pred_001 = ridge_001.predict(x_test)

plot_coef(x_train.columns, ridge_100.coef_)
plot_coef(x_train.columns, ridge_001.coef_)
## 어떤 하이퍼파라미터값을 주느냐에 따라 웨이트(w : 가중치)값이 다르게 나온다.
"""
학습데이터가 방대하여 실서비스를 내놓으려 할 때
모든 케이스를 커버할 수 있다면 규제를 많이 주지 않아도 된다.
그러나, 데이터분석 대회나 기업에서 고객 데이터, 매출액 데이터를 많이 보유하고 있지 않을 때
일반화를 시키려 한다면 규제를 많이 주는것이 좋다.
우리가 가진 데이터가 일반적인 사례를 모두 커버하지 못하기 때문에 우리 사례에만 과대적합된다면
실제로 출시했을 때 제대로 된 퍼포먼스가 나오지 못한다.
"""

## Lasso(라쏘)
from sklearn.linear_model import Lasso

for alpha in alphas:            # 각 규제 강도별로 모델 성능이 어떻게 나오는지
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    pred = lasso.predict(x_test)
    mse_eval('Lasso(alpha={})'.format(alpha), pred, y_test)         # 규제를 많이 줬을 때 퍼포먼스가 엄청나게 안좋게 나타난다.

lasso_100 = Lasso(alpha=100)
lasso_100.fit(x_train, y_train)
lasso_pred_100 = lasso_100.predict(x_test)

lasso_001 = Lasso(alpha=0.001)
lasso_001.fit(x_train, y_train)
lasso_pred_001 = lasso_001.predict(x_test)

plot_coef(x_train.columns, lasso_100.coef_)
plot_coef(x_train.columns, lasso_001.coef_)


# ElasticNet
"""
l1_ratio(default=0.5)
    l1_ratio = 0(L2 규제만 사용한다는 의미)
    l1_ratio = 1(L1 규제만 사용한다는 의미)
    0 < l1_ratio < 1 (L1 과 L2 규제의 혼합 사용)
"""
from sklearn.linear_model import ElasticNet
ratios = [0.2, 0.5, 0.8]

for ratio in ratios:
    elasticnet = ElasticNet(alpha=0.5, l1_ratio=ratio)
    elasticnet.fit(x_train, y_train)
    pred = elasticnet.predict(x_test)
    mse_eval('ElasticNet(l1_ratio={})'.format(ratio), pred, y_test)

elasticnet_20 = ElasticNet(alpha=5, l1_ratio=0.2)
elasticnet_20.fit(x_train, y_train)
elasticnet_pred_20 = elasticnet_20.predict(x_test)

elasticnet_80 = ElasticNet(alpha=5, l1_ratio=0.8)
elasticnet_80.fit(x_train, y_train)
elasticnet_pred_80 = elasticnet_80.predict(x_test)

plot_coef(x_train.columns, elasticnet_20.coef_)
plot_coef(x_train.columns, elasticnet_80.coef_)


# Scaler
## 스케일링을 해주느냐 안해주느냐, 어떤 스케일러로 해주느냐에 따라 모델 성능의 차이가 난다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## StandardScaler : 평균(mean)을 0, 표준편차(std)를 1로 만들어주는 스케일러
std_scaler = StandardScaler()
std_scaler = std_scaler.fit_transform(x_train)
print(round(pd.DataFrame(std_scaler).describe(), 2))

## MinMaxScaler : min값과 max값을 0~1 사이로 정규화
minmax_scaler = MinMaxScaler()
minmax_scaler = minmax_scaler.fit_transform(x_train)
print(round(pd.DataFrame(minmax_scaler).describe(), 2))

## RobustScaler : 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환
## outlier 값 처리에 유용
robust_scaler = RobustScaler()
robust_scaler = robust_scaler.fit_transform(x_train)
print(round(pd.DataFrame(robust_scaler).describe(), 2))


# 파이프라인
## 매번 스케일링하고 다시 넣어주고 하는 과정이 복잡한데 그 과정을 빠르게 처리할 수 있게 하는 방법
from sklearn.pipeline import make_pipeline

elasticnet_pipeline = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))      # standardscaler이나 ElasticNet뿐만 아니라 다른 값들도 사용 가능하다.

elasticnet_pred = elasticnet_pipeline.fit(x_train, y_train).predict(x_test)
mse_eval('Standard ElasticNet', elasticnet_pred, y_test)


# Polynomial Features
"""
다항식의 계수간 상호작용을 통해 새로운 feature르 생성한다.
예를들면, [a, b] 2개의 feature가 존재한다고 가정하고
degree=2 로 설정한다면, polynomial features는 [1, a, b, a^2, ab, b^2]가 된다.
"""
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x_train)[0]

print(poly_features[0])
print(x_train.iloc[0])

poly_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))
poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_test)
