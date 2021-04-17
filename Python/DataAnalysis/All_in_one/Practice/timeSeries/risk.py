
"""
# 어떤 투자가 잘한것일까?
주식 투자 30% 이익
벤쳐 캐피탈 30% 이익
비트코인 30% 이익

주식 => 변동성 : 0.1 => 0.3/0.1 = 3
벤쳐 캐피탈 => 변동성 : 0.3 => 0.3/0.3 = 1
비트코인 => 변동성 : 0.9 => 0.3/0.9 = 0.33

- 주식이 비트코인보다 약 9배 잘한 투자이다.
이러한 수치를 Sharpe ratio 라고 한다.

# 투자를 하기 위해 고려해야 하는 기본적인 요소 3가지
- 예상(초과) 수익률 : (안전 자산 대비) 얼마나 추가적인 수익률을 얻을 수 있는가?
- 위험 : 자산 가격이 얼마나 변동성이 있는가? -> 시계열의 표준편차
- Sharpe Ratio : 예상 초과 수익률 / 위험

# 리스크
- 리스크는 return의 표준편차로 나타낼 수 있다.

# Sharpe ratio : 단위 리스크당 수익
- risk adjusted return

# 여러가지 상품에 투자할 경우 포트폴리오를 어떻게 하느지에 따라 총 수익과 변동성이 달라진다.
ex) 상품 A : 수익률 10%, 변동성 10%
    상품 B : 수익률 20%, 변동성 20%
    
    상품 A:B = 3:7
        수익률 17%
        변동성 16%
        수익률/변동성 = 1.06
    상품 A:B = 5:5
        수익률 15%
        변동성 14%
        수익률/변동성 = 1.07

# 수익률은 3:7이 높지만 변동성 차이로 인해 Sharpe ratio 차이가 있다.
따라서 리스크를 고려했을 경우 5:5 투자가 더 수익을 얻을 확률이 높다.
이러한 방식으로 포트폴리오를 분석하여 투자한다.
"""

import pandas as pd
import numpy as np
import yfinance as yf       # 금융정보를 가져올 수 있는 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize 

"""
# 종목코드 예시
- 미국기업 : 'MSFT'
- 한국기업 : '005930.KS'    (삼성)
- 인덱스 : '^KS11'

# 포트폴리오 평가
- 여러 조합으로 자산 포트폴리오를 구성한 뒤에 Return, Risk, Sharpe Ratio 비교 평가
- 미국 Tech기업, 한국의 Tech기업의 투자 포트폴리오 비교해보기
"""

# 데이터 가져오기, 시각화
Tech_US = ['MSFT', 'NFLX', 'FB', 'AMZN']    # 마이크로소프트, 넷플릭스, 페이스북, 아마존
Tech_KR = ['005930.KS', '000660.KS', '035420.KS', '035720.KS']  # 삼성, SK하이닉스, 네이버, 카카오

print(yf.Ticker('MSFT').history(start='2019-04-01', end='2021-03-31'))     # 한달동안의 정보 / start, end로 기간을 지정할 수 있다.

# price(종가), dividends(배당)을 가져오는 함수를 정의
def get_price(companies):
    df = pd.DataFrame()
    for company in companies:
        df[company] = yf.Ticker(company).history(start='2019-04-01', end='2021-03-31')['Close']
    return df

def get_div(companies):
    df = pd.DataFrame()
    for company in companies:
        df[company] = yf.Ticker(company).history(start='2019-04-01', end='2021-03-31')['Dividends']
    return df

p_US = get_price(Tech_US)
print(p_US)

p_KR = get_price(Tech_KR)
p_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']

d_US = get_div(Tech_US)
d_KR = get_div(Tech_KR)
d_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']
# 2년간 기업당 배당금
print(d_US.sum())
print(d_KR.sum())

# 시각화
## 최초 가격 대비 변동률
(p_US/p_US.iloc[0]).plot(figsize=(15, 5))
plt.show()

(p_KR/p_KR.iloc[0]).plot(figsize=(15, 5))
plt.show()

# Daily Return
r_US = p_US / p_US.shift() - 1
r_KR = p_KR / p_KR.shift() - 1

# Average Return(Total period)
# 2년 평균 수입률
r_a_US = (p_US.iloc[-1] + d_US.sum()) / p_US.iloc[0] - 1
print(r_a_US)

r_a_KR = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
print(r_a_KR)

# Average Return(Daily)
r_a_d_US = (1+r_a_US)**(1/p_US.shape[0])-1
print(r_a_d_US)

r_a_d_KR = (1+r_a_KR)**(1/p_KR.shape[0])-1
print(r_a_d_KR)

# 포트폴리오 Returns
## 전체 포트폴리오 수익 : weights를 0.25, 0.25 0.25, 0.25로 투자한 경우를 가정
weights = np.array([0.25, 0.25, 0.25, 0.25])

port_return_US = np.dot(weights, r_a_d_US)
port_return_KR = np.dot(weights, r_a_d_KR)

print(port_return_US)
print(port_return_KR)

# 포트폴리오 Risk
## 
covar_US = r_US.cov() * 252
covar_KR = r_KR.cov() * 252

sns.heatmap(covar_US, cmap='PuBuGn')
plt.show()
## 넷플릭스와 페이스북의 리스크가 가장 크다
## 넷플릭스는 페이스북과 궁합이 잘 맞고 아마존은 페이스북과 궁합이 잘 맞는다
## 등 위와 같은 인사이트를 도출할 수 있따.

sns.heatmap(covar_KR, cmap='PuBuGn')
plt.show()

port_risk_US = np.dot(weights.T, np.dot(covar_US, weights))
port_risk_KR = np.dot(weights.T, np.dot(covar_KR, weights))

print(port_risk_US)
print(port_risk_KR)
## 리스크는 국내 투자가 조금 더 낮다.

# Sharpe ratio
rf = 0.02       # 미국 안전자산의 수익률(적금 등)
port_sr_US = (port_return_US - rf) / port_risk_US
port_sr_KR = (port_return_KR - rf) / port_risk_KR

print(port_sr_US)
print(port_sr_KR)
# 단위 리스크당 수익은 US 투자가 더 크다.

# 시각화
## 시각화를 위해 

result = np.array([[port_return_KR, port_return_US], [port_risk_KR, port_risk_US], [port_sr_KR, port_sr_US]])
result = np.round(result, 3)        # 소수점 3자리로 자른다.

result = pd.DataFrame(result)
result.columns = ['KR', 'US']
result.index = ['Return', 'Risk', 'Sharpe Ratio']
print(result)

result.plot(kind='bar')
plt.show()

"""
투자 상품의 포트폴리오 문제
- 나에게 있는 1억원을 어떻게 분배하여 투자할 것인가?
    부동산, 주식, 국채, 코인, 달러, 금 등

- 먼저 여러개 투자 상품의 수익과 리스크를 생각해본다.

covariance matrix

- 포트폴리오 최적화(Optimization) 예시 1
    개별 상품 조건
        상관관계 없는 똑같은 상품 2개 가정
        R1 = R2, 리스크가 같고 분산도 같음
    이러한 경우 1/2 일 때 최소 risk
    시사점 : 분산 투자 해야함

- 포트폴리오 최적화(Optimization) 예시 2
    개별 상품 조건
        똑같은 상품 2개 가정, 상관관계 = 1
        R1 = R2, 리스크가 같고 분산도 같음
    뭘 선택해도 같음
    시사점 : 유사한 상품으로 구성하지 말 것

- 포트폴리오 최적화(Optimization) 예시 3
    개별 상품 조건
        똑같은 상품 2개 가정, 상관관계 = -1
        R1 = R2, 리스크가 같고 분산도 같음
    시사점 : 성격이 반대인 상품으로 구성하면 좋음

- Capital Allocation Line(자본 분배선)
    안전자산과 위험자산의 분배선

Capital Allocation과 Efficient Frontier의 접점을 찾는것이 Portfolio
"""

# 최적화 기초 개념
"""
- 목적함수(Objective Function)
    최대 또는 최소로 달성하고자 하는 바
    ex) 매출

- 선택 변수(Choice variable)
    목적함수를 최대 또는 최소로 만들기 위하여 선택 가능한 변수(보통 weight라고 보면 된다.)
    ex) X1상품, X2상품의 생산량

- 제약조건(Constraint condition)
    목적함수를 달성하기 위하여 주어진 조건들
    ex) X1상품과 X2상품을 생산하는데 소요되는 재료

- 경계조건(Boundary condition)
    ex) X1, X2는 0보다 작을 수 없음

# 최적화 기초 문제
    선택변수 X1, X2 결정        arg max f(X1, X2)

Objective(목적)
    X1, X2의 제품의 판매 이익은 각각 40원, 30원이다
    식 : f(X1, X2) = 40 * X1 + 30 * X2

Constraint(제약)
    A원료는 X1제품과 X2제품 생산에 각각 4Kg, 5Kg이 소요되며, 총 재고는 50Kg이다.
    식 : 4 * X1 + 5 * X2 <= 50
    (inequality 조건이 있을수도 있다.)

    X1 제품과 X2 제품은 2:1의 비율로 생산해서 납품해야 한다.
    식 : 2 * X1 = X2
    (equality 조건이 있을수도 있다.)

Boundary(경계)
    X1, X2는 음의 값을 가질 수 없다.
    식 : X1, X2 >=0
"""

# 최적화 문제 정의
"""
목적함수 : Sharpe ratio(max), Risk(min)
선택변수 : weights
제약(constraint) : 모든 weights의 합은 1
한계(boundary) : 각 weight는 0과 1 사이
"""

# 포트폴리오 최적화 : 세가지 점을 구해봄
## 포트폴리오 리스크 최소
## Sharpe 지수 최대
## 효율적 투자점 : 목표 수익을 달성하기 위한 최소 risk를 가질 수 있는 포트폴리오

# minimize(목적함수, w0, constraints = , bounds=)

# 목적함수 정의
## 먼저 weights를 넣으면, return, risk, sharpe ratio를 return하는 함수를 정의 => 목적함수 정의
def get_stats(weights):
    r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
    port_return = np.dot(weights, r_a)
    covar_KR = (p_KR / p_KR.shift() - 1).cov() * 252
    port_risk = np.dot(weights.T, np.dot(covar_KR, weights))
    port_sharpe = port_return / port_risk
    return [port_return, port_risk, port_sharpe]

print(get_stats(weights))

def objective_return(weights):
    return -get_stats(weights)[0]

def objective_risk(weights):
    return get_stats(weights)[1]

def objective_sharpe(weights):
    return -get_stats(weights)[2]

# w0 정의
w0 = np.ones(len(Tech_KR)) / len(Tech_KR)

# constraints
constraints = {'type':'eq', 'fun':lambda x : np.sum(x)-1}

# bounds
bound = (0, 1)
bounds = tuple(bound for ii in range(len(Tech_KR)))

# 최적화 1. Risk 최소
opt_risk = minimize(objective_risk, w0, constraints=constraints, bounds=bounds)
print(minimize(objective_risk, w0, constraints=constraints, bounds=bounds))
"""
     fun: 0.06469435580685386                                           # minimize된 리스크가 0.06 이라는 의미
     jac: array([0.12929158, 0.1303496 , 0.12956896, 0.12950648])       
 message: 'Optimization terminated successfully.'
    nfev: 54
     nit: 9
    njev: 9
  status: 0
 success: True
       x: array([5.98590012e-01, 1.95156391e-18, 1.73924053e-01, 2.27485935e-01])       # 리스크가 최소화된 값(삼성, sk하이닉스, 네이버, 카카오 순)
"""
# 따라서 opt_risk['fun'] : 최적화된 리스크, opt_risk['x'] : 최적화된 때의 weights(포트폴리오)  이런 방식으로 출력한다.

# 최적화 2. Sharpe ratio 최대
opt_sharpe = minimize(objective_sharpe, w0, constraints=constraints, bounds=bounds)
print(-opt_sharpe['fun'])       # 최적화된 shapre ratio
print(opt_sharpe['x'])          # 그때의 weights(포트폴리오)

# 이러한 것들을 portfolio.py 의 내용처럼 scatter 그래프로 출력하면 시각화 할 수 있다.
# Ch04.주식종목 분석 및 포트폴리오 구성하기 - 05.(실습) 최적의 포트폴리오 도출하기.ipynb 참조

# 효율적 투자점 : 목표 수익을 달성하기 위한 최소 risk를 가질 수 있는 포트폴리오
target_returns = np.linspace(0.14, 0.23, 50)

target_risks = []
target_port = {}

for target_return in target_returns:
    constraints=({'type':'eq', 'fun':lambda x : np.sum(x)-1}, {'type':'eq', 'fun':lambda x : get_stats(x)[0]-target_return})
    opt_target=minimize(objective_risk, w0, constraints=constraints, bounds=bounds)
    target_risks.append(opt_target['fun'])
    target_port[target_return]=opt_target['x']

target_risks = np.array(target_risks)

print(target_risks)
w = pd.DataFrame(target_port.values())
w.columns = ['SS', 'SKH', 'NVR', 'KKO']
w.index = target_returns.round(3)
## 왼쪽에 있는 수익률을 달성하기 위한 best portfolio

w.plot(figsize=(12, 6), kind='bar', stacked=True)
plt.show()

"""
plt.scatter(port_risks, port_returns, c=port_returns/port_risks)
plt.colorbar(label='Sharpe ratio')
plt.xlabel('expected_risk')
plt.ylabel('expected_return')
plt.grid(True)
plt.show()
"""

