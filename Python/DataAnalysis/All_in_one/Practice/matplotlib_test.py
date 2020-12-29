import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colab에서의 한글 폰트 깨짐 현상 해결방법
"""
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
그 후 상단 메뉴 - 런타임 - 런타임 다시 시작 클릭

plt.rc('font', family='NanumBarunGothic')
"""

# 단일 그래프
data = np.arange(1, 100)
plt.plot(data)
plt.show()

# 다중 그래프
data = np.arange(1, 51)
data2 = np.arange(51, 101)

plt.plot(data)
plt.plot(data2)
plt.plot(data2 + 50)

plt.show()

# 2개의 figure로 나누어서 다중 그래프 그리기
## figure()는 새로운 크래프 canvas를 생성
plt.plot(data)
plt.figure()
plt.plot(data2)
plt.show()

# 여러개의 plot을 그리는 방법(subplot) : subplot(row, column, index)
data = np.arange(100, 201)
plt.subplot(2, 1, 1)
plt.plot(data)

data2 = np.arange(200, 301)
plt.subplot(2, 1, 2)
plt.plot(data2)

plt.show()

"""
# 세로로 나열하고자 할 때
data = np.arange(100, 201)
plt.subplot(1, 2, 1)
plt.plot(data)

data2 = np.arange(200, 301)
plt.subplot(1, 2, 2)
plt.plot(data2)

plt.show()
"""
# subplot에서 콤마가 없이도 표현이 가능하다.
data = np.arange(100, 201)
plt.subplot(211)
plt.plot(data)

data2 = np.arange(200, 301)
plt.subplot(212)
plt.plot(data2)

plt.show()

# 여러개의 plot을 그리는 방법(subplots) -> s가 더 붙는다.
# plt.subplots(행의 갯수, 열의 갯수)
data = np.arange(1, 51)

# 밑 그림
fig, axes = plt.subplots(2, 3)

axes[0, 0].plot(data)
axes[0, 1].plot(data * data)
axes[0, 2].plot(data ** 3)
axes[1, 0].plot(data % 10)
axes[1, 1].plot(-data)
axes[1, 2].plot(data // 20)

plt.tight_layout()
plt.show()

