# 미디어별 광고비와 세일즈 데이터를 가지고 최적의 마케팅 믹스를 구해본다.

"""
# 데이터 설명
    TV - TV 매체비
    radio - 라디오 매체비
    newspaper - 신문 매체비
    sales - 매출액

# 문제 정의
    - 전제
        실제로는 광고 매체비 이외의 많은 요인이 매출에 영향을 미친다.(영업인력 수, 입소문, 경기, 유행 등)
        이번 분석에서는 다른 요인이 모두 동일한 상황에서 매체비만 변경했을 때 매출액의 변화가 발생한 것이라고 간주해본다.
        실제로 Acquisition 단계에서는 종속변수가 매출액보다는 방문자수, 가입자수, DAU, MAU 등의 지표가 될 것이다.
        현재 2011년에 있다고 가정한다.
    - 분석의 목적
        각 미디어별로 매체비를 어떻게 쓰느냐에 따라서 매출액이 어떻게 달라질지 예측한다.
        궁극적으로는 매출액을 최대화할 수 있는 미디어 믹스의 구성을 도출한다.
        이 미디어믹스는 향후 미디어 플랜을 수립할 때 사용될 수 있다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Advertising.csv')
print(df.shape)
print(df.tail())
print(df.info())        # info는 결측치를 확인할 때 자주 사용한다.

## 분석에 필요한 컬럼만 선택한다.
df = df[['TV', 'radio', 'newspaper', 'sales']]

## 기술통계를 확인해본다.
print(df.describe())

## 변수간의 correlation을 확인한다.
print(df.corr())
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

## 변수간의 pairplot을 확인한다.
sns.pairplot(df[['TV', 'radio', 'newspaper', 'sales']])
plt.show()

## Labels와 features를 지정해준다.(Labels는 종속변수라 생각하면 된다.)
Labels = df["sales"]
features = df[['TV', 'radio', 'newspaper']]


# 데이터 분석
## 미디어별 매체비 분포를 scatterplot으로 시각화해본다.
figure, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(16, 6)

sns.scatterplot(data=df, x='TV', y='sales', ax=ax1)
sns.scatterplot(data=df, x='radio', y='sales', ax=ax2)
sns.scatterplot(data=df, x='newspaper', y='sales', ax=ax3)
plt.show()
# 매출액과의 scatter plot을 보면 TV가 매출액과 가장 관련이 높은 것 같아 보인다. 라디오도 관련이 있지만 신문의 상관관계는 애매해 보인다.

# 선형회귀 분석(stats model)
## stats model의 ols를 사용하면 선형회귀분석을 한다.
import statsmodels.formula.api as sm

model1 = sm.ols(formula='sales ~ TV + radio + newspaper', data=df).fit()        # formula에 종속변수 ~ 독립변수 를 적어준다. 

print(model1.summary())
"""
 OLS Regression Results
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.897
Model:                            OLS   Adj. R-squared:                  0.896
Method:                 Least Squares   F-statistic:                     570.3
Date:                Fri, 05 Feb 2021   Prob (F-statistic):           1.58e-96
Time:                        17:38:18   Log-Likelihood:                -386.18
No. Observations:                 200   AIC:                             780.4
Df Residuals:                     196   BIC:                             793.6
Df Model:                           3
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
TV             0.0458      0.001     32.809      0.000       0.043       0.049
radio          0.1885      0.009     21.893      0.000       0.172       0.206
newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
==============================================================================
Omnibus:                       60.414   Durbin-Watson:                   2.084
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
Skew:                          -1.327   Prob(JB):                     1.44e-33
Kurtosis:                       6.332   Cond. No.                         454.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
"""
- R-squared는 0.897로 매우 높은 편(물론 현실적으로는 이렇게 높게 나오기 쉽지 않다. 만약 데이터 분석을 할 때 너무 높게 나온다면 내 분석방법이 잘못되었는지도 의심해봐야 한다.)
    (즉, 종속변수하고 너무 연관성이 높은 변수가 존재할 가능성도 생각해봐야 한다.)
- P-value는 0.05 수준에서 유의한 변수는 TV, radio 이다.
    보통 0.05를 기준으로 잡고 0.05보다 작으면 유의미하다고 여기며 그 이상은 유의하지 않다고 본다.
- newspaper는 유의하지 않는 것으로 나타난다. 즉, 신문광고가 매출액에 미치는 영향은 유의하지 않다고 할 수 있다.
- 회귀식은 다음과 같다. sales = 2.9389 + 0.0458 TV + 0.1885 radio - 0.001*newspaper
- ceof는 타겟변수(즉, 종속변수)에 얼마나 영향을 미치는지를 나타내는 수치이다.
"""

# 선형회귀 분석(sklearn)
## sklearn의 선형회귀분석 결과도 같다.
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(features, Labels)
print("intercept_ : ", model.intercept_, " | coef : ", model.coef_)


## 변수의 포함여부에 따른 ols 결과도 확인해보자
model1 = sm.ols(formula='sales ~ TV + radio + newspaper', data=df).fit()
model2 = sm.ols(formula='sales ~ TV + radio', data=df).fit()
model3 = sm.ols(formula='sales ~ TV', data=df).fit()

print(model1.summary())
print(model2.summary())
print(model3.summary())
# 이렇게 하는 이유는 모델을 선택할 때 다양한 시나리오를 만들어 그려본 뒤 가장 성능이 좋은 모델을 선정하여 해당 모델을 쭉 가져가기 위함.
"""
- 유의하지 않은 변수 newspaper을 제거한 model2의 AIC, BIC가 가장 낮다.
- 여러개의 모델 중 선택할 때 AIC, BIC가 가장 낮은지 여부로 정하기도 한다.
- 물론 AIC, BIC가 유일한 판단기준은 아니고 RMSE, CFI 등 다른 기준들과 함께 고려되어야 한다.
- 결과에 따르면 p-value가 0.05 이상으로, 신문광고는 매출액 예측에 있어서 변수의 유무가 통계적으로 유의한 차이를 보이지 않는다.
- 즉, 신문광고 마케팅과 매출액은 관련이 없다고 할 수 있다.
"""

## 모델을 활용해 각 미디어별 매체비에 따른 sales를 예측해보자.
print(model1.predict({'TV' : 300, 'radio' : 10, 'newspaper' : 4}))
# 18.549433 이 출력되는데 예측한 sales값이 18.549433 이라 볼 수 있다.
# 즉, 위에서의 식처럼 sales = 2.9389 + 0.0458 * 300 + 0.1885 * 10 - 0.001 * 4

print(model2.predict({'TV' : 300, 'radio' : 10}))
# 단순하게 모델의 성능이 가장 좋았던 model2 (newspaper를 제거한)가 sales값이 가장 크게 출력된다.

# 데이터 변환 후 재분석
"""
- 신문광고가 유의미하지 않다고 나왔지만 데이터의 문제일 수도 있다는 생각이 들었다.
- 만약 2011년에 살고 있다고 가정하고, 상사는 여전희 신문광고가 유의미하다고 생각하고 있다.
- 분석결과에 대해 상사로부터 데이터 샘플수가 적거나 데이터 처리가 잘못되어 이런 결과가 나온 것이 아니냐는 지적을 받았다.
"""

# 데이터를 하나하나 따로 시각화해보니 newspaper 값이 치우쳐져 있다는 것을 알게 되었다.(소액 광고가 많았음.)
figure, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(16, 6)

sns.distplot(df['TV'], ax=ax1)
sns.distplot(df['radio'], ax=ax2)
sns.distplot(df['newspaper'], ax=ax3)
plt.show()

# 정규화를 위해 로그 변환을 해준다.
df['log_newspaper'] = np.log(df['newspaper'] + 1)     # 로그 변환을 할 때 그냥 해주면 0일 경우 음의 무한대값이 되므로 1을 더해준다.

figure, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(16, 6)

sns.distplot(df['TV'], ax=ax1)
sns.distplot(df['radio'], ax=ax2)
sns.distplot(df['log_newspaper'], ax=ax3)
plt.show()

# 변환한 newspaper 변수 결과도 포함하여 ols 분석을 진행해본다.
model4 = sm.ols(formula='sales ~ TV + radio + log_newspaper', data=df).fit()
print(model4.summary())

"""
결과 해석
    신문에 대한 상관계수는 음에서 양으로 변했지만 여전히 P-value 값이 0.05 수준에서 유의하지 않다.
    newspaper는 유의하지 않는 것으로 나타났다. 즉, 신문광고가 매출액에 미치는 옇양은 유의하지 않다고 할 수 있다.

총 해석
    상사는 여전히 신문광고가 유효하다고 생각하지만 데이터 분석결과는 그렇지 않다.
    신문광고를 중단하고 TV, 라디오 광고 위주로 집행해야 한다.
    그런데 TV광고는 비용대비 효율은 조금 떨어지는 것 같다.(coef가 radio보다 낮음)
    라이도 광고의 상관계수가 더 크다. 우리 제품은 라디오 광고를 할 수록 잘 팔리는 제품이다.
"""