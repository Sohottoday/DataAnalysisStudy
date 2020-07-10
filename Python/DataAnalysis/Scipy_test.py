# SciPy

# SciPy의 서브 패키지(서브 모듈)

## scipy.optimize : 최적화
## scipy.stats : 통계
## scipy.interplate : 보간법
## scipy.io : 데이터 입출력
## scipy.linalg : 선형 대수
## scipy.sparse : 희소 행렬
## scipy.special : 특수 수학 함수
## scipy.signal : 신호 처리
## scipy.cluster : 벡터 양자화
## scipy.constants : 물리/수학 상수
## scipy.integrate : 통합
## scipy. ndimage : n차원 이미지 패키지
## scipy.spatial : 공간 데이터 구조 및 알고리즘
## scipy.odr : 직교 거리 회귀
## scipy.fftpack : 푸리에 변환

# 확률 분포 객체를 다루기 위한 명령
## 이산 확률 분포(베르누이 분포, 이항 분포, 다항 분포)
## 연속 확률 분포(정규 분포, 균등 분포...)

# stats 서브 패키지 안에 포함된 명령
## 이산 : bernoulli(베르누이 분포), binom(이항 분포), multinomial(다항 분포)
## 연속 : uniform(균일 분포), norm(가우시안 정규 분포), beta(베타 분포), gamma(감마 분포),
##      t(스튜던트 t분포), chi2(카이제곱 분포), f(F분포), dirichlet(디리클리 분포), multivariate_normal(다변수 가우시안 정규분포)

import scipy as sp
from scipy import stats

rv = sp.stats.norm()         # 정규분포 객체 rv를 생성
print(type(rv))

# 모수 지정
## 확률 분포 객체를 생성할 때는 분포의 형태를 구체적으로 지정하는 모수(parameter)를 인수로 지정해야 한다.
## loc(분포의 기대값), scale(표준편차) 두 개의 모수는 대부분 공통적으로 사용한다.
## size(샘플 생성시 생성될 샘플의 크기), random_state(샘플 생성 시 사용되는 seed값)
rv = sp.stats.norm(loc=1, scale=2)      # 기대값이 1이고 표준편차가 2인 정규분포 객체 생성

# 확률 분포 메서드
## pdf() : 확률 밀도 함수(probability density function)
## pmf() : 확률 질량 함수(probability mass function)
## cdf() : 누적 분포 함수(cumulative distribution function)
## rvs() : 랜덤 샘플 생성(random variable sampling)

# 확률 밀도 함수
import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-6, 6, 100)
pdf = rv.pdf(x)
plt.plot(x, pdf)
plt.grid(True)
plt.show()

# 누적 분포 함수
cdf = rv.cdf(x)
plt.plot(x, cdf)
plt.grid()
plt.show()

# 랜덤 샘플 생성(rvs 메서드를 활용한다. size, random_state 모수를 사용한다.)
rv.rvs(size=(2, 4), random_state=0)
print(rv.rvs(size=(2, 4), random_state=0))

import seaborn as sns
sns.distplot(rv.rvs(size=10000, random_state=0))
plt.xlim(-6, 6)
plt.show()


# 실수 분포 plot (seaborn 패키지 : distplot, kdeplot, rugplot)



# 베르누이 분포
# 베르누이 시도(Bernoulli trial) : 결과가 성공 또는 실패 두가지 중 하나로만 나오는 것
# 동전을 던져서 앞이나 뒤가 나오는 경우 베르누이 시도라 한다.

# 베르누이 시도 결과를 확률 변수(random variable) X로 나타낼 때는 일반적으로 성공은 1로 실패는 0으로 표현한다.
# 가끔 실패를 -1로 정하는 경우도 있다.
# 베르누이 확률 변수는 0, 1 두 가지 값 중 하나만 가질 수 있으므로 이산 확률 변수이다.
# 1이 나올 확률을 theta(모수)로 표현하며, 0이 나올 확률은 1-0으로 표현한다.

# scipy를 이용한 베르누이 분포
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# scipy에서 베르누이 분포의 모수 설정을 p 속성으로 사용한다.
theta = 0.6     # 성공할 확률이 0.6이라는 의미, 즉 실패 확률은 0.4
rv = sp.stats.bernoulli(theta)

# pmf 메서드를 이용하면 확률 질량 함수(pmf : probability mass function)를 계산할 수 있다.

xx = [0, 1]
plt.bar(xx, rv.pmf(xx))

plt.xticks([0, 1], ['X=0', 'X=1'])
plt.show()

# 베르누이 분포 시뮬레이션
x = rv.rvs(100, random_state=0)
print(x)

sns.countplot(x)
plt.show()

y = np.bincount(x, minlength=2) / float(len(x))
df = pd.DataFrame({'theory' : rv.pmf(xx), 'simulation' : y})
df.index = [0, 1]
print(df)

df2 = df.stack().reset_index()
df2.columns = ['sample value', 'type', '%']
print(df2)

sns.barplot(x='sample value', y='%', hue='type', data=df2)      # 한글 인식 X
plt.show()


# seaborn barplot 메서드는 범주 내에 있는 통계 측정


# Scipy를 이용한 이항 분포 시뮬레이션
## scipy의 서브 모듈 binom 클래스가 이항 분포 클래스이다.
## n과 p 속성을 사용하여 모수를 설정한다.

N = 10
theta = 0.6
rv = sp.stats.binom(N, theta)

xx = np.arange(N+1)
plt.bar(xx, rv.pmf(xx))
plt.ylabel('pmf(x)')
plt.show()

# 시뮬레이션 (rvs 메서드 사용)
x = rv.rvs(100)
print(x)

sns.set()
sns.set_style('darkgrid')
sns.countplot(x)
plt.show()


y = np.bincount(x, minlength=N+1) / float(len(x))
df = pd.DataFrame({'theory' : rv.pmf(xx), 'simulation' : y}).stack()
df = df.reset_index()
df.columns = ['sample', 'type', '%']
df.pivot('sample','type','%')
print(df)

sns.barplot(x='sample', y='%', hue='type', data=df)
plt.show()