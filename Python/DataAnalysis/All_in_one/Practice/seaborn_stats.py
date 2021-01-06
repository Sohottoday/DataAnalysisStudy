import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

sns.set(style='darkgrid')

titanic = sns.load_dataset('titanic')
tips = sns.load_dataset('tips')
print(tips.head())

# Countplot
sns.countplot(x='class', hue='who', data=titanic, palette='Paired')
plt.show()

## 위의 그래프를 가로로 보고 싶을 땐 x가 아닌 y값으로 지정해주면 된다
sns.countplot(y='class', hue='who', data=titanic)
plt.show()


# distplot
## hist 그래프와 kdeplot을 통합한 그래프 : 분포와 밀도를 확인할 수 있다.
x = np.random.randn(100)

sns.distplot(x)
plt.show()

## 데이터가 Series일 경우
x = pd.Series(x, name='x variable')
sns.distplot(x)
plt.show()          # X축의 이름이 column 이름으로 자동 설정된다.


## rugplot
## 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 데이터들의 위치 및 분포를 보여준다.
sns.distplot(x, rug=True, hist=False, kde=True)
plt.show()

## kde(kernel density) : histogram보다 부드러운 형태의 분포 곡선을 보여주는 방법
sns.distplot(x, rug=False, hist=False, kde=True)
plt.show()

## 가로로 표현하고자 할 때
# sns.distplot(x, vertical=True)

## 색을 바꾸고자 할 때
# sns.distplot(x, color='y')


# heatmap
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data, annot=True)           # annot을 False로 주면 히트맵 위의 숫자가 없어진다.
plt.show()

## pivot table을 활용하여 그리기
pivot = tips.pivot_table(index='day', columns='size', values='tip')
print(pivot)

sns.heatmap(pivot, cmap='Blues', annot=True)
plt.show()

## correlation(상관관계)를 시각화 하기 좋음 
### corr() 함수는 데이터의 상관관계를 보여준다.
print(titanic.corr())

sns.heatmap(titanic.corr(), annot=True, cmap='YlGnBu')
plt.show()


# pairplot : grid 형태로 각 집합의 조합에 대해 히스토그램과 분포도를 그려준다. 또한, 숫자형 column에 대해서만 그려준다.
## 한번에 여러가지 정보를 얻을 수 있어 데이터 분석 시작 전에 찍어보면 대략적인 느낌을 알 수 있다.
sns.pairplot(tips)
plt.show()

## hue 옵션으로 특성 구분, palette 옵션으로 색상 지정, height로 사이즈 적용
sns.pairplot(tips, hue='size', palette='rainbow', height=5)
plt.show()


# violinplot
## 바이올린처럼 생긴 plot
## column에 대한 데이터의 비교 분포도를 확인할 수 있다.
## 양쪽 끝 뾰족한 부분은 데이터의 최소값과 초대값을 나타낸다.
sns.violinplot(x=tips['total_bill'])
plt.show()

## x, y 축을 지정해줌으로써 바이올린을 분할하여 비교 분포를 볼 수 있다.
sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()
## 다른 그래프와 마찬가지로 가로로 된 그래프를 보려면 x값과 y값을 바꿔주면 된다.

## 사실 hue 옵션을 사용하지 않으면 바이올린이 대칭이기 때문에 비교 분포의 큰 의미는 없다.
## 하지만 hue 옵션을 주면, 단일 column에 대한 바이올린 모 양의 비교를 할 수 있다.
## split 옵션으로 바이올린을 합쳐서 볼 수 있다.(비교할 때 용이하다.)
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, palette='muted', split=True)
plt.show()

# lmplot
## column간의 선형관계를 확인하기에 용이한 차트, 또한 outlier도 같이 집작해 볼 수 있다.
sns.lmplot(x='total_bill', y='tip', height=8, data=tips)
plt.show()

## hue 옵션으로 다중 선형관계 그리기
## 아래의 그래프를 통해 비흡연자가, 흡연자 대비 좀 더 가파른 선형관계를 가지는 것을 볼 수 있다.
sns.lmplot(x='total_bill', y='tip', hue='smoker', height=8, data=tips)
plt.show()

## col 옵션을 추가하여 그래프를 별도로 그려볼 수 있다. 또한 col_wrap으로 한 줄에 표기할 column의 갯수를 명시할 수 있따.
sns.lmplot(x='total_bill', y='tip', hue='smoker', height=4, data=tips, col_wrap=2)
plt.show()


# relplot
## 두 column간 상관관계를 보지만 lmplot처럼 선형관계를 따로 그려주지는 않는다.
## col 옵션으로 그래프 분할이 가능하다.
## row와 column에 표기할 데이터 column 선택
sns.relplot(x='total_bill', y='tip', hue='day', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='day', col='time', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='day',row='sex', col='time', data=tips)
plt.show()


# jointplot
## scatter(산점도)와 histogram(분포)을 동시에 그려준다.
## 숫자형 데이터만 표현 가능하다.
sns.jointplot(x='total_bill', y='tip', height=8, data=tips)
plt.show()

## 선형관계를 표현하는 regression 라인 그리기 : 옵션에 kind='reg'를 추가해준다.
sns.jointplot(x='total_bill', y='tip', height=8, data=tips, kind='reg')
plt.show()

## kind='hex' 옵션을 통해 분포를 더 시각적으로 확인할 수 있다.
sns.jointplot(x='total_bill', y='tip', height=8, data=tips, kind='hex')
plt.show()

## 등고선 모양으로 밀집도 확인하기 : kind='kde'
sns.jointplot(x='total_bill', y='tip', height=8, data=tips, kind='kde', color='g')
plt.show()
