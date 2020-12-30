import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

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


# 타이틀 설정
plt.plot(np.arange(10), np.arange(10)*2)
plt.plot(np.arange(10), np.arange(10)**2)
plt.plot(np.arange(10), np.log(np.arange(10)))

plt.title('이것은 타이틀 입니다', fontsize=20)

# X축 & Y축 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# label 각도 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

# 범례 설정(legend)
plt.legend(['10*2', '10**2', 'log'], fontsize=15)

# X와 Y의 한계점(limit) 설정 -> xlim(), ylim()
# plt.xlim(0, 5)
# plt.ylim(0.5, 10)

plt.show()

"""
[세부 도큐먼트 확인하기](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot) 

**marker의 종류**
* '.'	point marker
* ','	pixel marker
* 'o'	circle marker
* 'v'	triangle_down marker
* '^'	triangle_up marker
* '<'	triangle_left marker
* '>'	triangle_right marker
* '1'	tri_down marker
* '2'	tri_up marker
* '3'	tri_left marker
* '4'	tri_right marker
* 's '	square marker
* 'p'	pentagon marker
* '*'	star marker
* 'h'	hexagon1 marker
* 'H'	hexagon2 marker
* '+'	plus marker
* 'x'	x marker
* 'D'	diamond marker
* 'd'	thin_diamond marker
* '|'	vline marker
* '_'	hline marker


**line의 종류**
* '-' solid line style
* '--' dashed line style
* '-.' dash-dot line style
* ':' dotted line style


**color의 종류**
* 'b'	blue
* 'g'	green
* 'r'	red
* 'c'	cyan
* 'm'	magenta
* 'y'	yellow
* 'k'	black
* 'w'	white

** alpha**
투명도 설정

"""

plt.plot(np.arange(10), np.arange(10)*2, marker='o', linestyle='', color='b')
plt.plot(np.arange(10), np.arange(10)*2 - 10, marker='o', linestyle='-', color='c', alpha=0.3)
plt.plot(np.arange(10), np.arange(10)*2 - 20, marker='v', linestyle='--', color='y', alpha=0.6)
plt.plot(np.arange(10), np.arange(10)*2 - 30, marker='+', linestyle='-.', color='y', alpha=1.0)
plt.plot(np.arange(10), np.arange(10)*2 - 40, marker='*', linestyle=':')

plt.title('다양한 선의 종류 예제',fontsize=20)

plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

plt.xticks(rotation=90)
plt.yticks(rotation=30)

# grid 옵션 추가
plt.grid()

plt.show()




