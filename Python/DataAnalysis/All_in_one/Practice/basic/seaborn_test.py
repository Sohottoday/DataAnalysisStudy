# seaborn : matplotlib을 기반으로 다양한 색상과 차트를 지원하는 라이브러리

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import Image

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

# seaborn을 선호하는 이유
## seaborn에서만 제공되는 통계 기반 plot
## seaborn의 최대 장점 중 하나인 컬러 팔렛트
## hue 옵션으로 bar를 구분 - xtick, ytick, xlabel, ylabel, legend까지 자동으로 생성, 신뢰도 구간도 알아서 계산한 뒤 생성해 준다.
## col 옵션 하나로 그래프 자체를 분할해 준다.
tips = sns.load_dataset("tips")

sns.violinplot(x='day', y='total_bill', data=tips)
plt.title('violin plot')
plt.show()


# matplotlib 차트를 seaborn에서 구현
## scatter
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.arange(50)
area = x * y * 1000

"""
plt.scatter(x, y, s=area, c=colors)
"""
### seaborn에서는 size와 sizes를 동시에 지정해준다.
### sizes 옵션에서는 사이즈의 min, max를 명시해 준다.
### hue는 컬러 옵션
### palette를 통해 seaborn이 제공하는 palette를 사용
sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), hue=area, palette='coolwarm')
plt.show()

### 여러개의 그래프를 출력하는 방식도 비슷하다.
plt.figure(figsize=(12, 6))

plt.subplot(131)
sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), color='blue', alpha=0.1)
plt.title('alpha=0.1')

plt.subplot(132)
sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), color='red', alpha=0.5)
plt.title('alpha=0.5')

plt.subplot(133)
sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), color='green', alpha=0.9)
plt.title('alpha=0.9')

plt.show()

## barplot
x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]
"""
plt.bar(x, y, align='center', alpha=0.7, color='red')
plt.xticks(x)
plt.ylabel('Scores')
plt.title('Subjects')
plt.show()
"""

sns.barplot(x, y, alpha=0.8, palette='YlGnBu')

plt.ylabel('Score')
plt.title('Subject')
plt.show()

### seaborn에서 barh를 표현하고자 할 때에는 x와 y값을 바꿔주면 된다.
sns.barplot(y, x, alpha=0.8, palette='YlGnBu')

plt.ylabel('Score')
plt.title('Subject')
plt.show()

## seaborn에서 비교그래프 그리기
### seaborn에서 비교그래프는 matplotlib와 다른 방식을 취한다.
### tip : 그래프를 임의의 데이터로 그려야 하는 경우 - matplotlib
###       DataFrame을 가지고 그리는 경우 - seaborn
titanic = sns.load_dataset('titanic')
print(titanic.head())

sns.barplot(x='sex', y='survived', hue='pclass', data=titanic, palette='muted')
plt.show()


# lineplot
x = np.arange(0, 10, 0.1)
y = 1 + np.sin(x)

## grid 스타일도 설정할 수 있다  - whitegrid, darkgrid, white, dark, ticks
sns.set_style("darkgrid")

sns.lineplot(x, y)

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin graph', fontsize=18)

plt.show()


# Area plot
## seaborn은 지원하지 않음


# Histogram - distplot
N = 100000
bins = 30
x = np.random.randn(N)

sns.distplot(x, bins=bins, kde=False, hist=True, color='g')
plt.show()
## kde를 False로 설정하면 데이터의 갯수가 Y축에 표기된다.
## kde를 True로 설정해주면, Density가 Y축에 표기 된다.(밀도라고 생각하면 된다.)
sns.distplot(x, bins=bins, kde=True, hist=True, color='g')
plt.show()

## vertical 옵션을 True로 주면 그래프를 수직으로 꺽는다.
sns.distplot(x, bins=bins, kde=True, hist=True, vertical=True, color='g')
plt.show()


# pie chart
## Seaborn에서는 pie chart를 지원하지 않음


# Box plot
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

## orient : 박스플롯을 어떻게 볼 것인지 -> 아무 속성을 주지 않으면 가로로, 'v' 속성을 주면 세로로 보여준다.
## width : 각 박스 플롯의 너비를 설정해 준다.
sns.boxplot(data, orient='v', width=0.2)
plt.show()

## 다중 box plot
sns.boxplot(x='pclass', y='age', hue='survived', data=titanic)
plt.show()

