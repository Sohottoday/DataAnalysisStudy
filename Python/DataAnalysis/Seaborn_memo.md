# Seaborn 

- 데이터 분포 시각화 페이지

- 실수 분포 플롯(distplot, kdeplot, rugplot)
- import seaborn as sns로 주요 사용



- seaborn의 스타일 지정
- set 명령으로 색상, 틱, 전체적인 플롯 스타일을 Seaborn 스타일로 바꿀 수 있다.
- set_style은 틱 스타일만 바꿀 수 있다. darkgrid, whitegrid, dark, white, ticks 스타일 제공

- set_color_codes : 기본 색상을 바꿀 수 있다.



- 사용 순서

  `sns.set()`

  `set_style('whitegrid')`

``` python
import matplotlib.pyplot as plt
import seaborn as sns
```



- 색상 팔렛트(deep, muted, pastel, dark, bright, colorblind)

```python
cur_pal = sns.color_palette()
sns.palplot(cur_pal)
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/58559786/87035432-49bdad00-c224-11ea-90a6-039776c24da3.png)

``` python
sns.palplot(sns.color_palette('Blues'))
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/58559786/87035526-6d80f300-c224-11ea-8a62-945a2bd4e290.png)

``` python
sns.palplot(sns.color_palette('bright', 10))
plt.show()
```

![Figure_2](https://user-images.githubusercontent.com/58559786/87035618-8d181b80-c224-11ea-9afe-9b8dca6dba27.png)



- 붓꽃 데이터 로드

``` python
iris = sns.load_dataset('iris')
```



- rugplot : rug(작은 선분)을 이용하여 x축 위에 실제 데이터들의 위치를 보여주는 plot

``` python
data = iris.petal_length.values
sns.set()
sns.set_style('whitegrid')
sns.rugplot(data)
plt.show()
```

![Figure_3](https://user-images.githubusercontent.com/58559786/87035800-d23c4d80-c224-11ea-8d6b-d7f5363695af.png)

- kdeplot(kernel density plot) : 커널 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포곡선을 보여주는 plot

``` python
sns.kdeplot(data)
plt.show()
```

![Figure_4](https://user-images.githubusercontent.com/58559786/87035994-1af40680-c225-11ea-9d2e-aab482560699.png)



- distplot 
  - 러그 표시와 커널 밀도 표시 기능이 있어서 matplotlib에서 제공하는 hist 명령보다 더 많이 사용된다.
  - kde = False 설정하면 히스토그램으로 표시 가능

``` python
sns.distplot
plt.show()

sns.distplot(data, bins=20, kde=False, rug=True)		# bins는 막대 분할 개수
plt.show()
```

![Figure_5](https://user-images.githubusercontent.com/58559786/87036122-4aa30e80-c225-11ea-95b2-0a8dffaf4d6b.png)

- seaborn을 이용한 1차원 데이터 분포 표시

  카운트 플롯

  - countplot() : 각 카테고리 값 별로 데이터가 얼마나 있는지를 알 수 있는 플롯
  - countplot 명령은 데이터프레임에서만 사용할 수 있다.
  - count(x = 'column_name', data=dataframe)

``` python
titanic = sns.load_dataset('titanic')		# 타이타닉 데이터 로드

sns.countplot(x = 'class', data=titanic)
plt.show()
```

![Figure_6](https://user-images.githubusercontent.com/58559786/87036372-b1282c80-c225-11ea-9856-17b50fad00f4.png)







