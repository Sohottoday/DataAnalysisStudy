import numpy as np

# Sum / Product

# sum : 여러개의 수를 연속하여 더하는 연산(그리스 문자 시그마)
# product : 여러개의 수를 연속하여 곱하는 연산(그리스 문자 파이)

# 선형 대수(linear algebra) : 데이터 분석에 필요한 여러 계산을 돕기 위한 학문

# 데이터 유형 : scalar, vector, matrix
# scalar(스칼라) : 숫자 하나로 이루어져 있는 데이터
# vector(벡터) : 여러개의 숫자로 이루어진 데이터(data record)
# matrix(행렬) : 벡터가 여러개 있는 데이터 집합

# numpy에서 벡터를 표현할 때 벡터를 열의 개수가 하나인 2차원 배열 객체로 표현하는 것이 올바르다.
x1 = np.array([[2.1], [2.2], [2.3]])        # numpy로 벡터를 표현
# numpy에서는 1차원 배열 객체도 대부분 벡터로 인정한다.
x1 = np.array([2.1, 2.2, 2.3])      # 위와 같이 벡터로 인정되지만 출력할 때 가로로 출력된다.

# matrix : 하나의 데이터 레코드를 단독으로 벡터로 나타낼때는 하나의 열로 나타낸다.
# 복수의 데이터 레코드 집합을 행렬로 표현할 때는 하나의 데이터 레코드가 행으로 표현

# 전치 연산 : 행렬에서 가장 기본이 되는 연산. 행과 열을 바꾸는 연산
# 보통 T로 표현한다.
# 전치 연산으로 만들어진 행렬을 원래 행렬에 대한 '전치행렬' 이라고 한다.
# 벡터 x에 대한 전치 연산을 적용하여 만든 xT는 행의 수가 1인 행렬이므로 행 벡터(row vector)라고 한다.
x1 = np.array([[2.2], [3.3], [4.4], [5.5]])
print(x1)

# T는 메서드가 아니라 속성이므로 소괄호()를 붙이지 않는다.
print(x1.T)

x2 = np.array([1, 2, 3, 4])
print(x2)
print(x2.T)     # 1차원 ndarray는 전체 연산이 정의되지 않는다.

# 행렬의 행 표기와 열 표기

# 특수 벡터(0벡터, 1벡터)
## 0벡터는 모든 성분이 0으로만 구성된 벡터이며 1벡터는 1로만 구성된 벡터

# 특수 행렬(정방 행렬, 대각 행렬)
## 정방 행렬(square matrix) : 행의 개수와 열의 개수가 같은 행렬
## 대각 행렬 : numpy의 diag명령으로 생성 가능
## 단위 행렬(identity matrix) : 대각행렬 중에서 모든 대각 성분의 값이 1인 대각행렬, 대문자 I로 표현된다.
i = np.identity(3)
print(i)

i2 = np.eye(3)
print(i2)

## 대칭 행렬(symmetric matrix) : 전치 연산을 통해 얻은 전치 행렬과 원래의 행렬이 같은 경우. 단, 정방행렬만이 될 수 있다.


# Numpy를 이용한 벡터의 기하학적 의미

# 벡터를 N차원 공간상에서 표현을 할 때 점(point) 또는 화살표(arrow)로 표현할 수 있다.

# 차트상에서 주석 처리하는 함수 : annotate(s, xy, xytext, xycoords, arrowprops, ....)
## s : 주석, xy : 화살표 시작, xytext : 주석 텍스트 시작
import matplotlib.pyplot as plt

ax = plt.subplot(1, 1, 1)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
line = plt.plot(t, s, lw = 2)

plt.annotate('max value', xy=(3, 1), xytext=(4, 1.5), arrowprops=dict(facecolor='red'))
plt.ylim(-2, 2)
plt.show()

a = np.array([1, 3])
plt.plot(0, 0, 'kP', ms=20)     # ms는 marker의 size
plt.xticks(np.arange(-2, 5))
plt.yticks(np.arange(-1, 5))
plt.plot(a[0], a[1], 'bo', ms=20)

plt.annotate('', xy=a, xytext=(0, 0), arrowprops=dict(facecolor='black'))
plt.text(0.2, 1.5, "$a$", fontdict={'size':20})

plt.xlim(-2.5, 4.4)
plt.ylim(-0.6, 4.4)
plt.show()

# 벡터의 길이
# 2차원 벡터 a의 길이는 피타고라스 정리를 이용하여 얻을 수 있는데, 그 값을 벡터의 놈(norm)이라고 한다.
# 수학 표기법은 ||a|| 이다.
# numpy에서는 linalg 서브 패키지에 norm 함수를 이용해서 벡터의 길이를 알 수 있다.

a = np.array([1, 3])
alength = np.linalg.norm(a)
print(alength)


# 스칼라와 벡터의 곱
# 양수와 벡터를 곱하면 벡터의 방향은 변하지 않고 그 양수의 크기만큼 벡터의 크기가 커진다.
# 음수를 곱하면 벡터의 방향은 반대방향이 된다.

import numpy as np
import matplotlib.pyplot as plt


a = np.array([1, 2])
b = 2 * a
c = -1 * a

plt.annotate('', xy=b, xytext=(0,0), arrowprops=dict(facecolor='red'))
plt.text(0.9, 3.2, '$2a$', fontdict={'size':15})

plt.annotate('', xy=a, xytext=(0,0), arrowprops=dict(facecolor='black'))
plt.text(0.2, 1.2, "$a$", fontdict={"size" : 15})

plt.annotate('', xy=c, xytext=(0,0), arrowprops=dict(facecolor='yellow'))
plt.text(-0.2, -0.7, '$-a$', fontdict={'size':15})
plt.plot(c[0], c[1], 'go', ms=15)

plt.plot(0, 0, 'ko', ms=20)
plt.xticks(np.arange(-4, 6))
plt.yticks(np.arange(-4, 6))
plt.xlim(-4.5, 5.5)
plt.ylim(-3.5, 5.5)
plt.grid(True)
plt.show()

# 단위 벡터(unit vector) : 길이가 1인 벡터
a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
avalue = np.linalg.norm(a)
bvalue = np.linalg.norm(b)
cvalue = np.linalg.norm(c)

print(avalue, '//', bvalue, '//', cvalue)

# 벡터의 합
## 두 벡터를 이웃하는 변으로 가지는 평행사변형의 다른 쪽 모서리가 두 벡터를 합한 벡터의 위치이다.
a = np.array([1, 2])
b = np.array([2, 1])
c = a + b   # a 벡터와 b벡터를 합한 c도 역시 벡터가 된다

plt.annotate('', xy=a, xytext=(0,0), arrowprops=dict(facecolor='black'))
plt.annotate('', xy=b, xytext=(0,0), arrowprops=dict(facecolor='black'))
plt.annotate('', xy=c, xytext=(0,0), arrowprops=dict(facecolor='yellow'))

plt.plot(0, 0, 'bo', ms=15)
plt.plot(a[0], a[1], 'bo', ms=15)
plt.plot(b[0], b[1], 'bo', ms=15)
plt.plot(c[0], c[1], 'bo', ms=15)

plt.plot([a[0],c[0]], [a[1], c[1]], 'r--' )
plt.show()

# 기하학적인 또다른 벡터의 합 표현
plt.annotate('', xy=a, xytext=(0, 0), arrowprops=dict(facecolor='blue'))
plt.annotate('', xy=c, xytext=a, arrowprops=dict(facecolor='blue'))
plt.annotate('', xy=c, xytext=(0, 0), arrowprops=dict(facecolor='red'))

plt.plot(0, 0, 'go', ms=10)
plt.plot(a[0], a[1], 'go', ms=10)
plt.plot(c[0], c[1], 'go', ms=10)

plt.text(0.4, 1.2, "$a$", fontdict={'size':15})
plt.text(1.4, 2.5, '$b$', fontdict={'size':15})
plt.text(1.3, 1.5, '$c$', fontdict={'size':15})

plt.xticks(np.arange(-2, 5))
plt.yticks(np.arange(-1, 4))
plt.show()


