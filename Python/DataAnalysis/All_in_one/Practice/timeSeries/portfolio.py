import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

Tech_KR =['005930.KS','000660.KS','035420.KS', '035720.KS' ] #삼성, SK하이닉스, 네이버, 카카오   

def get_price(companies): 
  df=pd.DataFrame()
  for company in companies:
    df[company]=yf.Ticker(company).history(start='2019-04-01', end='2021-03-31')['Close']
  return df

def get_div(companies): 
  df=pd.DataFrame()
  for company in companies:
    df[company]=yf.Ticker(company).history(start='2019-04-01',end='2021-03-31')['Dividends']
  return df  

p_KR=get_price(Tech_KR)
d_KR=get_div(Tech_KR)

p_KR.columns=['SS', 'SKH', 'NVR', 'KKO']
d_KR.columns=['SS', 'SKH', 'NVR', 'KKO']

# 3000개의 임의의 weights를 생성해서 return, risk를 도시
weights = np.random.rand(len(Tech_KR))
weights = weights/np.sum(weights)       # 가중치는 다 더해서 1이 되야 하므로 이러한 방식으로 설정해둔다.

# 임의의 생성된 난수에 대한 Return
r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1     # 마지막 가격과 배당을 더한 값을 시작 가격으로 나눠준다.
port_return = np.dot(weights, r_a)      # 포트폴리오 리턴은 11%

# 임의의 생성된 난수에 대한 Risk
r_d = p_KR / p_KR.shift() - 1
covar_KR = (p_KR / p_KR.shift() - 1).cov() * 252
port_risk = np.dot(weights.T, np.dot(covar_KR, weights))
print(port_risk)

# weights의 조합에 따른 포트폴리오 리턴, 리스크
port_returns = []
port_risks = []

for ii in range(3000):
    weights = np.random.rand(len(Tech_KR))
    weights = weights / np.sum(weights)
    r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
    port_return = np.dot(weights, r_a)
    covar_KR = (p_KR / p_KR.shift() - 1).cov() * 252
    port_risk = np.dot(weights.T, np.dot(covar_KR, weights))

    port_returns.append(port_return)
    port_risks.append(port_risk)

# 임의의 weight에 대하여 return들과 risk들을 얻을 수 있다.
# 위 값들은 행렬계산 해야하기 때문에 array로 바꿔준다.
port_returns = np.array(port_returns)
port_risks = np.array(port_risks)

plt.scatter(port_risks, port_returns, c=port_returns/port_risks)
plt.colorbar(label='Sharpe ratio')
plt.grid(True)
plt.xlabel('expected_risk')
plt.ylabel('expected_return')
plt.show()
# x축이 리스크 y축이 리턴
## 리스크가 적고 리턴이 많은것을 찾아본다.
## 그래프를 통해 Sharpe ratio도 찾을 수 있다.

