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



# 앙상블(Ensemble)
"""
# 머신러닝 앙상블이란 여러개의 머신러닝 모델을 이용해 최적의 답을 찾아내는 기법
    여러 모델을 이용하여 데이터를 학습하고, 모든 모델의 예측결과를 평균하여 예측
# 앙상블 기법의 종류
    - 보팅(Voting) : 투표를 통해 결과 도출
    - 배깅(Bagging) : 샘플 중복 생성을 통해 결과 도출
    - 부스팅(Boosting) : 이전 오차를 보완하면서 가중치 부여
    - 스태킹(Stacking) : 여러 모델을 기반으로 예측된 결과를 통해 meta 모델이 다시 한번 예측
"""

# 보팅(Voting)
## 보팅(Voting) - 회귀(Regression)
"""
# Voting은 말 그대로 투표를 통해 결정하는 방식
# Voting은 Bagging과 투표방식이라는 점에서 유사하지만, 다음고 같은 큰 차이점이 있다.
    Voting은 다른 알고리즘 model을 조합해서 사용한다.
    Bagging은 같은 알고리즘 내에서 다른 sample 조합을 사용한다.
"""
from sklearn.ensemble import VotingRegressor, VotingClassifier

## 반드시, Tuple 형태로 모델을 정의해야 한다.
single_models = [
    # ('linear_reg', linear_reg),
    ('ridge', ridge),
    ('lasso', lasso),
    ('elasticnet_pipeline', elasticnet_pipeline),
    ('poly_pipeline', poly_pipeline)
]

voting_regressor = VotingRegressor(single_models, n_jobs=-1)
voting_regressor.fit(x_train, y_train)
voting_pred = voting_regressor.predict(x_test)
mse_eval('Voting Ensemble', voting_pred, y_test)

## 보팅(Voting) - 분류(Classification)
"""
분류기 모델을 만들때, Voting 앙상블은 1가지의 중요한 parameter가 있다.
voting = {'hard', 'soft'}
VotingClassifier(voting='soft')

# hard로 설정한 경우
    class를 0, 1로 분류 예측을 하는 이진 분류를 예로 들어
    Hard Voting 방식에서는 결과 값에 대한 다수 class를 차용한다.
    classification을 예로 들어 보자면, 분류를 예측한 값이 1, 0, 0, 1, 1 이었다고 가정한다면 1이 3표, 0이 2표를 받았기 때문에
    Hard Voting 방식에서는 1이 최종 값으로 예측하게 된다.

# soft로 설정한 경우
    soft vote 방식은 각각의 확률의 평균 값을 계산한 다음에 가장 확률이 높은 값으로 확정짓게 된다.
    가령 class 0이 나올 확률이 (0.4, 0.9, 0.9, 0.4, 0.4)이었고, class 1이 나올 확률이 (0.6, 0.1, 0.1, 0.6, 0.6) 이었따면
    class 0이 나올 최종 확률은 (0.4, 0.9, 0.9, 0.4, 0.4)/5 = 0.44
    class 1이 나올 최종 확률은 (0.6, 0.1, 0.1, 0.6, 0.6)/5 = 0.4
    가 되기 때문에 앞선 Hard Vote의 결과와는 다른 결과 값이 최종으로 선출되게 된다.

# 보통 soft 방식을 조금 더 선호한다.
"""


# 배깅(Bagging)
"""
Bagging은 Bootstrap Aggregating 의 줄임말
    Bootstrap = Sample(샘플) + Aggregating = 합산
    즉, 여러개의 dataset을 중첩을 허용하게 하여 샘플링하여 분할하는 방식

    데이터 셋의 구성이 [1, 2, 3, 4, 5]로 되어 있다면
    1. group 1 = [1, 2, 3]
    2. group 2 = [1, 3, 4]
    3. group 3 = [2, 3, 5]

# 대표적인 Bagging 앙상블
    RandomForest
    Bagging
"""

## RandomForest
"""
DecisionTree(트리) 기반 Bagging 앙상블
굉장히 인기있는 앙상블 모델
사용성이 쉽고, 성능도 우수

# 주요 Hyperparameter
    - random_state : 랜덤 시드 고정 값. 고정해두고 튜닝할 것
    - n_jobs : CPU 사용 갯수
    - max_depth : 깊어질 수 있는 최대 깊이. 과대적합 방지용
    - n_estimators : 앙상블하는 트리의 갯수, default값은 100개
    - max_features : 최대로 사용할 feature의 갯수. 과대적합 방지용
    - min_sample_split : 트리가 분할할 때 최소 샘플의 갯수. default=2. 과대적합 방지용
"""
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=7, max_features=0.8)
rfr.fit(x_train, y_train)

rfr_pred = rfr.predict(x_test)
mse_eval('RandomForest', rfr_pred, y_test)


# 부스팅(Boosting)
"""
# 약한 학습기를 순차적으로 학습을 하되, 이전 학습에 대해 잘못 예측된 데이터에 가중치를 부여해 오차를 보완해나가는 방식
    장점 : 성능이 우수하다(Lgbm, XGBoost)
    단점 : 부스팅 알고리즘의 특성상 계속 약점(오분류/잔차)을 보완하려고 하기 때문에 잘못된 레이블링이나 아웃라이어에 필요 이상으로 민감할 수 있다.
        다른 앙상블 대비 학습 시간이 오래 걸린다

# 대표적인 Boosting 앙상블
1. AdaBoost
2. GradientBoost
3. LightGBM(LGBM)
4. XGBoost
"""

## GradientBoost
### 성능이 우수하다
### 학습시간이 해도해도 너무 느리다.
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.8)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
mse_eval('GradientBoost Ensemble', gbr_pred, y_test)

"""
# 주요 Hyperparameter
    - random_state : 랜덤 시드 고정 값. 고정해두고 튜닝할 것
    - n_jobs : CPU 사용 갯수
    - learning_rate : 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야 함. default = 0.1
    - n_estimators : 부스팅 스테이지 수, default값은 100개
    - subsample : 샘플 사용 비율(max_features와 비슷한 개념). 과대적합 방지용
    - min_sample_split : 노드 분할시 최소 샘플의 갯수. default=2. 과대적합 방지용

보통 learning_rate와 n_estimators는 같이 움직인다.
"""

## XGBoost : eXtreme Gradient Boosting
"""
주요 특징
    scikit-learn 패키지가 아니다.
    성능이 우수하다
    GBM보다 빠르고 성능도 향상되었다.
    여전히 학습시간이 매우 느리다.
"""
from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.8, max_features=0.8, max_depth=7)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
mse_eval('XGB Ensemble', xgb_pred, y_test)
"""
# 주요 Hyperparameter
    - random_state : 랜덤 시드 고정 값. 고정해두고 튜닝할 것
    - n_jobs : CPU 사용 갯수
    - learning_rate : 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야 함. default = 0.1
    - n_estimators : 부스팅 스테이지 수, default값은 100개
    - max_depth : 트리의 깊이. 과대적합 방지용. default = 3
    - subsample : 샘플 사용 비율(max_features와 비슷한 개념). 과대적합 방지용
    - max_features : 최대로 사용할 feature의 비율. 과대적합 방지용. default=1.0

보통 learning_rate와 n_estimators는 같이 움직인다.
scikit-learn 패키지가 아니므로 GPU버전으로 설치한다면 GPU 사용도 가능하다.
"""

## LightGBM
"""
주요 특징
    scikit-learn 패키지가 아니다.
    성능이 우수하다
    속도도 매우 빠르다.
"""
from lightgbm import LGBMRegressor, LGBMClassifier

lgbm = LGBMRegressor(random_state=42, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.8, subsample=0.8, max_depth=7)
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
mse_eval('LGBM Ensemble', lgbm_pred, y_test)
"""
# 주요 Hyperparameter
    - random_state : 랜덤 시드 고정 값. 고정해두고 튜닝할 것
    - n_jobs : CPU 사용 갯수
    - learning_rate : 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야 함. default = 0.1
    - n_estimators : 부스팅 스테이지 수, default값은 100개
    - max_depth : 트리의 깊이. 과대적합 방지용. default = 3
    - colsample_bytree : 샘플 사용 비율(max_features와 비슷한 개념). 과대적합 방지용. default=1.0

보통 learning_rate와 n_estimators는 같이 움직인다.
scikit-learn 패키지가 아니므로 GPU버전으로 설치한다면 GPU 사용도 가능하다.
"""


# Stacking
"""
# 개별 모델이 예측한 데이터를 기반으로 final_estimator 종합하여 예측을 수행한다.
    성능을 극으로 끌어올릴 때 활용한다.
    과대적합을 유발할 수 있다.(특히 데이터셋이 적은 경우)
    시간이 많이 소요된다.
"""
from sklearn.ensemble import StackingRegressor

stack_models = [
    ('elasticnet', poly_pipeline),
    ('randomforest', rfr),
    ('gbr', gbr),
    ('lgbm', lgbm),
]

stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=-1)
stack_reg.fit(x_train, y_train)
stack_pred = stack_reg.predict(x_test)
mse_eval('Stacking Ensemble', stack_pred, y_test)


## Weighted Blending
"""
각 모델의 예측값에 대하여 weight(가중치)를 곱하여 최종 output 계산
    모델에 대한 가중치를 조절하여, 최종 output을 산출한다.
    가중치의 합은 1.0이 되도록 한다.
"""

final_outputs = {
    'elasticnet' : poly_pred,
    'randomforest' : rfr_pred,
    'gbr' : gbr_pred,
    'xgb' : xgb_pred,
    'lgbm' : lgbm_pred,
    'stacking' : stack_pred,
}

final_prediction=\
final_outputs['elasticnet'] * 0.1\
+final_outputs['randomforest'] * 0.1\
+final_outputs['gbr'] * 0.2\
+final_outputs['xgb'] * 0.25\
+final_outputs['lgbm'] * 0.15\
+final_outputs['stacking'] * 0.2\
# 만약 이전의 결과에서 XGBoost가 결과가 좋았다면 xgboost에 가중치를 더 주는 방식과 같이 진행하면 된다.


# 앙상블 모델을 정리하며
"""
1. 앙상블은 대체적으로 단일 모델 대비 성능이 좋다.
2. 앙상블을 앙상블하는 기법인 Stacking과 Weighted Blending도 참고해 볼 만 하다.
3. 앙상블 모델은 적절한 Hyperparameter 튜닝이 중요하다.
4. 앙상블 모델은 대체적으로 학습시간이 더 오래 걸린다.
5. 따라서, 모델 튜닝을 하는 데에 걸리는 시간이 오래 소요된다.
"""


# Cross Validation
"""
Cross Validation 이란 모델을 평가하는 하나의 방법
K-겹 교차검증(K-fold Cross Validation)을 많이 활용한다.

# K-겹 교차검증
    K-겹 교차 검증은 모든 데이터가 최소 한 번은 테스트셋으로 쓰이도록 한다.
    ex) Estimation 이 1일때,
    학습데이터 : [B, C, D, E] / 검증데이터 : [A]
    Estimation 2일때,
    학습데이터 : [A, C, D, E] / 검증데이터 : [B]
    https://static.packt-cdn.com/products/9781789617740/graphics/b04c27c5-7e3f-428a-9aa6-bb3ebcd3584c.png

'CV 한다' 라는 의미는 교차검증 한다는 의미
"""
from sklearn.model_selection import KFold

n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=42)

X = np.array(df.drop('MEDV', 1))
Y = np.array(df['MEDV'])

lgbm_fold = LGBMRegressor(random_state=42)

i = 1
total_error = 0
for train_index, test_index in kfold.split(X):          # 5개로 split했기 때문에 5번 반복된다.
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    lgbm_pred_fold = lgbm_fold.fit(x_train_fold, y_train_fold).predict(x_test_fold)
    error = mean_squared_error(lgbm_pred_fold, y_test_fold)
    print('Fold = {}, prediction score = {:.2f'.format(i, error))
    total_error += error
    i+=1
print('---' * 10)
print('Average Error : %s' % (total_error / n_splits))


# Hyperparameter 튜닝
"""
    hyperparameter 튜닝시 경우의 수가 너무 많다.
    따라서, 우리는 자동화할 필요가 있다.

# sklearn 패키지에서 자주 사용되는 hyperparameter 튜닝을 돕는 클래스는 다음 2가지가 있다.
    1. RandomizedSearchCV
    2. GridSearchCV

# 적용하는 방법
    1. 사용할 Search 방법을 선택한다.
    2. hyperparameter 도메인을 설정한다.(max_depth, n_estimators 등등)
    3. 학습을 시킨 후 기다린다.
    4. 도출된 결과 값을 모델에 적용하고 성능을 비교한다.
"""

## RandomizedSearchCV
### 적절한 하이퍼파라미터를 찾기 위한 과정
"""
모든 매개 변수 값이 시도되는 것이 아니라 지정된 분포에서 고정 된 수의 매개 변수 설정이 샘플링된다.
시도 된 매개 변수 설정의 수는 n_iter(시도횟수)에 의해 제공된다.

# 주요 Hyperparameter(LGBM)
    - random_state : 랜덤 시드 고정 값. 고정해두고 튜닝할 것
    - n_jobs : CPU 사용 갯수
    - learning_rate : 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야 함. default = 0.1. n_estimators와 같이 튜닝
    - n_estimators : 부스팅 스테이지 수, default값은 100개
    - max_depth : 트리의 깊이. 과대적합 방지용. default = 3
    - colsample_bytree : 샘플 사용 비율(max_features와 비슷한 개념). 과대적합 방지용. default=1.0
"""

params = {
    'n_estimators' : [200, 500, 1000, 2000],
    'learning_rate' : [0.1, 0.05, 0.01],
    'max_depth' : [6, 7, 8],
    'colsample_bytree' : [0.8, 0.9, 1.0],
    'subsample' : [0.8, 0.9, 1.0],
}

from sklearn.model_selection import RandomizedSearchCV
# n_iter 값을 조절하여, 총 몇 회의 시도를 진행할 것인지 정의(횟수가 늘어나면, 더 좋은 parameter를 찾을 확률이 올라가지만 그만큼 시간이 오래걸린다.)

clf = RandomizedSearchCV(LGBMRegressor(), params, random_state=42, cv=3, n_iter=25, scoring='neg_mean_squared_error')
# cv : cross validation(위 참고),    scoring : ???
clf.fit(x_train, y_train)
clf.best_score_             # RandomizedSearchCV가 찾은 최적의 parameter들을 넣고 진행했을 때의 값
clf.best_params_            # RandomizedSearchCV가 찾은 각 parameter들의 최적의 값들을 출력해준다.


## GridSearchCV
### 모든 매개 변수값에 대하여 완전 탐색을 시도한다.
### 따라서, 최적화할 parameter가 많다면, 시간이 매우 오래 걸린다.
params = {
    'n_estimators' : [500, 1000],
    'learning_rate' : [0.1, 0.05, 0.01],
    'max_depth' : [7, 8],
    'colsample_bytree' : [0.8, 0.9],
    'subsample' : [0.8, 0.9],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(LGBMRegressor(), params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
grid_search.best_score_
grid_search.best_params_