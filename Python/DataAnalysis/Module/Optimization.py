
# 최적화(Optimization)
# 최적화의 문제 : 함수 f의 값을 최소화하는 변수 x의 값 x*을 찾는 것, 이때 x*을 최적화의 해라고 한다.

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x-2)**2 + 2

x1 = np.linspace(-1, 4, 100)        # -1부터 4까지 100개 뽑아냄

plt.plot(x1, f(x1))
plt.plot(2, 2, 'ro', ms=10)

plt.show()
# 최적화의 기본 개념


# 그리드 서치와 수치적 최적화
# 최적화 문제 : 목적 함수의 값을 가장 최소로 하는 x의 위치를 찾는 것
# 최적화 문제를 푸는 가장 간단한 방법 : x의 값을 여러번 넣어 보고 그 중에서 가장 작은 값을 선택하는 방법(grid search)
# 그리드 서치를 이용하면 목적함수의 값(예측 오차)을 찾기 위한 계산량이 상당히 많아진다.
# 목적함수의 계산량을 줄이기 위한 방법으로 수치적 최적화(numerical optimization)를 사용한다.
# 수치적 최적화 : 반복적 시행착오(trial and error)에 의해 최적화 필요조건을 만족하는 x*를 찾는 방법
# 함수의 위치가 최저점이 될 때까지 가능한 적은 횟수로 x의 위치를 옮기는 방법

# 수치적 최적화 방법에 필요한 알고리즘
    # 1. 어떤 위치 x를 시도한 뒤 에 다음번에 시도할 위치 x를 찾는 알고리즘
    # 2. 현재 위치 x가 최저점인지를 판단할 수 있는 알고리즘

# 기울기 필요 조건 : 모든 최소점은 기울기가 0이기 때문에 최소점이 되기 위해서는 기울기값이 0이어야 한다.
# 기울기가 0이어도 최소점이 아닐 수 있다.
# 기울기를 나타내는 벡터(함수)를 g(gradient)로 표현한다.

# SGD(Steepest Gradient Descent) 방법
## 현재 위치 x에서의 기울기 값 g(x)만을 이용해서 다음에 시도할 위치를 알아내는 방법


def f1(x):
    return (x-2)**2 + 2

def f1d(x):
    return 2 * (x - 2.0)


xx = np.linspace(-1, 6, 100)
plt.plot(xx, f1(xx), 'b-')

# step size 값 설정
mu = 0.4

# k 는 위치값, 0부터 시작한다. k = 0 첫번 위치 시도값
x = 0

plt.plot(x, f1(x), 'mo', ms=10)
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'g--')
print('x = {}, g = {}'.format(x, f1d(x)))



# k = 1 일 때
x = x - mu * f1d(x)

plt.plot(x, f1(x), 'co', ms=10)
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'g--')
print('x = {}, g = {}'.format(x, f1d(x)))

plt.ylim(0, 10)


# k = 2 세번째 시도
x = x - mu * f1d(x)

plt.plot(x, f1(x), 'ro', ms=10)
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'g--')
print('x = {}, g = {}'.format(x, f1d(x)))

plt.show()


# SciPy를 이용한 최적화
# import scipy as sp 로 불러온다
# scipy의 서브 패키지는 optimize에 minimize 메서드를 이용하여 최적화를 할 수 있다.
# 이 때 사용하는 알고리즘은 BFGS 이다.
# minimize 메서드는 최적화를 시작하기 위한 초기값을 인수로 사용한다. 목적함수를 인수로 사용한다.

# minimize 명령은 최적화 결과를 OptimizeResult 클래스 객체로 반환한다.
# OptimizeResult 클래스는 다음과 같은 속성을 갖고 있다.
# x : 최적화 해
# success : 최적화를 성공했을 때 True를 반환
# status : 종료 상태. 최적화에 성공했을 때 0을 반환
# message : 메세지 문자열
# fun : x위치에서의 함수 값
# jac : x 위치에서의 자코비안(그래디언트) 벡터 값
# hess : x 위치에서 헤시안 행렬 값
# nfev : 목적 함수를 몇번 호출했는지 횟수
# njev : 자코비안 계산 횟수
# nhev : 헤시안 계산 횟수
# nit : x 이동 횟수

from scipy import optimize
import scipy as sp

def f0(x):
    return (x - 2) ** 2 + 2

x0 = 0      # 초기값 설정
res = sp.optimize.minimize(f0, x0)
print(res)
