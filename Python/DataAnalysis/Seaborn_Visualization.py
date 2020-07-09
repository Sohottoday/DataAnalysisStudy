# Seaborn : 데이터 분포 시각화 패키지
# 실수 분포 플롯 (distplot, kdeplot, rugplot)
# import seaborn as sns 로 주로 사용

# seaborn의 스타일 지정
# set 명령으로 색상, 틱, 전체적인 플롯 스타일을 Seaborn 스타일로 바꿀 수 있다.
# set_style은 틱 스타일만 바꿀 수 있다. darkgrid, whitegrid, dark, white, ticks 스타일 제공

# set_color_codes : 기본 색상을 바꿀 수 있다.

# 사용 순서
# sns.set()
# set_style('whitegrid')

import matplotlib.pyplot as plt
import seaborn as sns


# 색상 팔렛트(deep, muted, pastel, dark, bright, colorblind)
cur_pal = sns.color_palette()
sns.palplot(cur_pal)
plt.show()

sns.palplot(sns.color_palette('Blues'))
plt.show()

sns.palplot(sns.color_palette('bright', 10))
plt.show()

# 붓꽃 데이터 로드
iris = sns.load_dataset('iris')

# rugplot : rug(작은 선분)을 이용하여 x축 위에 실제 데이터들의 위치를 보여주는 plot
data = iris.petal_length.values
sns.set()
sns.set_style('whitegrid')
sns.rugplot(data)
plt.show()

# kdeplot(kernel density plot) : 커널 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포곡선을 보여주는 plot
sns.kdeplot(data)
plt.show()

# distplot : 러그 표시와 커널 밀도 표시 기능이 있어서 matplotlib에서 제공하는 hist 명령보다 더 많이 사용되고 있다.
## kde = False 설정하면 히스토그램으로 표시 가능
sns.distplot
plt.show()

sns.distplot(data, bins=20, kde=False, rug=True)        # bins는 막대 분할 개수
plt.show()

# seaborn을 이용한 1차원 데이터 분포 표시

# 카운트 플롯
## countplot() : 각 카테고리 값 별로 데이터가 얼마나 있는지를 알 수 있는 플롯
## countplot 명령은 데이터프레임에서만 사용할 수 있다.

# 사용 예
# countplot(x = 'column_name', data=dataframe)

titanic = sns.load_dataset('titanic')   # 타이타닉 데이터 로드

sns.countplot(x = 'class', data = titanic)
plt.show()

