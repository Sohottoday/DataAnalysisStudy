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


# 카테고리 분포

# 카테고리 분포는 1부터 K까지의 K개의 정수 값 중 하나가 나오는 확률 변수의 분포이다.
# 예를 들면, 주사위를 던져 나오는 수를 확률 변수라고 가정하면, 확률변수는 {1, 2, 3, 4, 5, 6} 값이 나온다.
# 클래스의 수 K = 6 인 카테고리 분포라고 한다.

# 카테고리 분포에서는 벡터 확률 변수로 사용한다.
# X = 1 --> X = (1, 0, 0, 0, 0, 0)
# X = 2 --> X = (0, 1, 0, 0, 0, 0)
# X = 3 --> X = (0, 0, 1, 0, 0, 0)
# X = 4 --> X = (0, 0, 0, 1, 0, 0)
# X = 5 --> X = (0, 0, 0, 0, 1, 0)
# X = 6 --> X = (0, 0, 0, 0, 0, 1)

# 이러한 인코딩 방식을 One-Hot-Encoding 방식이라고 한다.

# scipy를 이용한 카테고리 분포 시뮬레이션

# scipy에서는 별도의 카테고리 분포 클래스를 제공하지 않는다. 다항 분포를 위한 클래스 multinomial를 이용하여
# 카테고리 분포 객체를 생성할 수 있다. multinomial 클래스에서 시행 횟수를 1로 설정하면
# 카테고리 분포가 되므로 이 클래스를 활용한다.

theta = np.array([1/6] * 6)         # 1/6 확률로 6개 설정, 6개가 각각 균등한 확률
rv = sp.stats.multinomial(1, theta)

xx = np.arange(1, 7)
# get_dummies() 는 One-Hot-Encoding 을 만들 때 사용하는 함수이다.
xx_ohe = pd.get_dummies(xx)
print(xx_ohe)

#plt.bar(xx, rv.pmf(xx_ohe))

#plt.ylabel('P(x)')
#plt.title('pmf of Categorical Distribution')
#plt.show()             에러가 왜 뜨는지 모르겠음

# 시뮬레이션
X = rv.rvs(100)
print(X[:10])

y = X.sum(axis=0) / float(len(X))
plt.bar(np.arange(1, 7), y)
plt.title('Simulation of Categorical distribution')
plt.show()

df = pd.DataFrame({'theory' : rv.pmf(xx_ohe), 'simulation':y}, index= np.arange(1, 7)).stack()
df = df.reset_index()
df.columns = ['sample', 'type', '%']
df.pivot('sample', 'type', '%')
print(df)       # 여기서도 같은 부분에 No axis named -1 for object type <class 'pandas.core.frame.DataFrame'> 오류 발생

eps = np.finfo(np.float).eps
theta = np.array([eps, eps, 0.2, 0.1, 0.4, 0.3])            # 6개라고 가정했을 때 각각의 확률
rv = sp.stats.multinomial(1, theta)

X = rv.rvs(100, random_state=1)
y = X.sum(axis=0) / float(len(X))               # 행끼리 더하므로 axis=0

df = pd.DataFrame({'theory' : rv.pmf(xx_ohe), 'simulation':y}, index=np.arange(1, 7)).stack()
df = df.reset_index()
df.columns = ['sample', 'type', '%']
df.pivot('sample', 'type', '%')
sns.barplot(x='sample', y='%', hue='type', data=df)     # hue : 카테고리 변수 이름을 지정하여 카테고리 값에 따라 색상을 다르게 함.
plt.show()


# 다항 분포(Multinomial distribution)
# 카테고리 분포의 확장

# scipy를 이용한 다항 분포의 시뮬레이션

# seaborn의 2차원 복합데이터를 표현하는 플롯 소개
# barplot
# pointplot
# boxplot : 중앙값, 표준 편차 등 간략한 특성을 보여주며, 박스는 실수 값 분포에서 1사분위수와 3사분위수를 의미
## 박스 내부에 있는 가로선은 중앙값을 의미하며 박스 외부의 세로선은 1.5 * IQR(Q3-Q1) 만큼 1사분위수보다 낮은 값과 
## 1.5 * IQR 만큼 3사분위수보다 높은 값의 구간을 기준으로 그 구간의 내부에 있는 가장 큰 값과 가장 작은 값을 이어준 선분
## 그 바깥에 있는 점은 아웃라이어(outlier)라고 한다.
# violinplot : 세로 방향으로 커널 밀도 히스토그램을 표현하며, 좌우가 대칭이 되어 바이올린 모양으로 보여진다.
# stripplot : scatter plot 처럼 모든 데이터를 점으로 표시한다. (jitter = True) => 데이터의 수가 많을 경우 겹치지 않게 설정
# swarmplot

tips = sns.load_dataset('tips')
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()

sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()

sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
plt.show()

sns.swarmplot(x='day', y='total_bill', data=tips)
plt.show()

# 다차원 복합 데이터

sns.boxplot(x='day', y='total_bill', hue='sex', data=tips)
plt.show()

sns.violinplot(x='day', y='total_bill', hue='sex', data=tips)
plt.show()

sns.stripplot(x='day', y='total_bill', hue='sex', data=tips, jitter=True)
plt.show()

sns.swarmplot(x='day', y='total_bill', hue='sex', data=tips)
plt.show()

sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, split=True)
plt.show()

sns.stripplot(x='day', y='total_bill', hue='sex', data=tips, dodge=True)
plt.show()

sns.swarmplot(x='day', y='total_bill', hue='sex', data=tips, dodge=True)
plt.show()

# scipy는 다항 분포를 위한 multinomial 클래스를 제공한다. 인수로는 시행 횟수 : N, theta

N = 30
theta = [0, 0, 0.1, 0.2, 0.3, 0.4]
rv = sp.stats.multinomial(N, theta)

X = rv.rvs(100)     # 샘플을 100개 얻어옴

print(X[:6])

plt.boxplot(X)
plt.show()

df = pd.DataFrame(X).stack().reset_index()
df.columns = ['trial', 'class', 'binomial']

sns.boxplot(x='class', y='binomial', data=df)
sns.stripplot(x='class', y='binomial', data=df, jitter=True, color='.2')
plt.show()

sns.violinplot(x='class', y='binomial', data=df, inner=None)        # inner = {'box', 'point', 'stick', 'None', 'quartile'}
sns.swarmplot(x='class', y='binomial', data=df, color='.2')
plt.show()

# 가우시안 정규 분포(Gaussian normal distribution)

mu = 0
std = 1
rv = sp.stats.norm(mu, std)

xx = np.linspace(-5, 5, 100)
plt.plot(xx, rv.pdf(xx))
plt.show()

x = rv.rvs(100)
print(x)

sns.distplot(x, kde=True, fit=sp.stats.norm)
plt.show()

# Q-Q 플롯 : 통계 분석 중의 하나로 정규분포검정(normality test)을 시각적으로 확인하는 plot
# Q-Q 플롯 사용 순서
## sample 데이터를 크기순으로 정렬
## 각 샘플 데이터의 분위함수(quantile function)값을 구한다.
## 분위수(quantile)를 구한다.
## 샘플 데이터와 그에 대응한 정규분포값을 하나의 쌍으로 생각하고 2차원 공간에 점으로 그린다.
## 모든 샘플에 대해 위의 과정을 반복하여 scatter plot을 완성한다.

# scipy에서는 Q-Q plot을 그리기 위한 명령 probplot 메서드 사용
# probplot : 기본적으로 속성을 통해 보낸 데이터 샘플에 대한 Q-Q plot 정보만을 반환하고 실제로 차트를 그리지는 않는다.
# plot 속성에는 matplotlib.pyplot 모듈 객체를 값으로 한다.

# 정규 분포를 따르는 데이터 샘플을 Q-Q plot으로 보여주기
x = np.random.randn(100)
plt.figure(figsize=(5,5))
sp.stats.probplot(x, plot=plt)
plt.axis('equal')
plt.show()


# 정규 분포를 따르지 않는 데이터 샘플을 Q-Q plot으로 보여주기
x = np.random.rand(100)
plt.figure(figsize=(5, 5))
sp.stats.probplot(x, plot=plt)
plt.ylim(-0.5, 1.5)
plt.show()

# scipy를 이용한 t분포 알아보기

# scipy에서는 t명령을 이용하여 t분포 객체를 생성한다.
# 사용되는 속성은 표준편차, 기대값, 자유도
# 자유도(degree of freedom)가 클수록 정규분포에 수렴된다.

xx = np.linspace(-4, 4, 100)

for df in [1, 3, 5, 10, 20]:
    rv = sp.stats.t(df=df)
    plt.plot(xx, rv.pdf(xx), label=('student-t(dof=%d)' % df))

plt.plot(xx, sp.stats.norm().pdf(xx), label='Normal', lw=4, alpha=0.5)     # aplha : 투명도
plt.legend()      # legend : 범례
plt.show()