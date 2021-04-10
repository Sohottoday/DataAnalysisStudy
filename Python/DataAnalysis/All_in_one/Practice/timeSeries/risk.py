
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