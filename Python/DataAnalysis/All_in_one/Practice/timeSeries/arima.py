
# ARMA : AR+MA(p, q)
"""
# ARMA(Auto Regressive Moving Average) 모델
- 예를들어, ARMA(2, 2) 모델은 AR(2) + MA(2) 모델임

# ARIMA(Auto Regressive Integrated Moving Average) 모델
- ARMA 모델의 원계열 Yt를 차분하여 Yt' 로 변환한 모형
- 예를 들어, ARIMA(2, 1, 2) 모델은 원계열을 1번 차분하고 AR(2) + MA(2)을 진행한 모델임

# ARIMA 모델을 적용할 때 주의점
- ARIMA(p, k, q)모델 : AR(p), Integrated(k), MA(q)
    라이브러리에 입력할 때 Yt를 입력하면, Stationary해질 때까지 Yt', Yt", ... Yt**k와 같이 차분
    차분하지 않는 경우, 설명력이 매우 높은 모형이 생성됨 -> Training 데이터에서만 정확도가 높은 잘못된 결과일 확률이 매우 높음
    시계열의 단순 차분값을 활용하기 보다 변동율로 변환하기 위해서 원계열에 log를 취하고나 △log(log difference)를 취하는 경우도 많음.
    원데이터        log(원데이터)       △log
    200            log(200)          NAN
    300            log(300)          log(300)-log(200)
    200            log(200)          log(200)-log(300)
    100            log(100)          log(100)-log(200)

# 예측 모형 만드는 순서
안정성 검토
    Yt가 안정적이지 않으면, △Yt가 안정적인지 확인 -> 보통 1~2번의 차분으로 안정적인 시계열이 됨(ARIMA(p, 1, q) 또는 ARIMA(p, 2, q) 선정)
데이터 특성에 맞는 모형 결정(AR차수와 MA차수)
    PACF peak와 ACF peak의 개수로 AR, MA계수 선정.
    PACF의 peak이 p개, ACF의 peak이 q개 이면, ARIMA(p, k, q) 모델 선정
학습
    특정 시점 이전 데이터(Training set)로 학습
평가
    특정 시점 이후 데이터(Test set)로 평가

- 다 지나간 학습데이터를 맞춰보는게 목적이 아니라면 테스트 데이터를 분리해서 사용해야 한다.
- AR 프로세스만 잘 학습해서 이전 관측치를 그 다음기에 그대로 예측하는 경향이 있다(딥러닝도 똑같음)
  '변동'을 예측하도록 보완하던지, 아예 Level보다는 변동을 예측(실무에서 주로 사용)
"""