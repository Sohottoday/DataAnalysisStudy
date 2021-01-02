import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'


# Scatter
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.arange(50)
area = x * y * 1000         # 점의 넓이. 값이 커지면 당연히 넓이도 커진다.

plt.scatter(x, y, s=area, c=colors)         # s : size라는 의미 / c : color라는 의미
plt.show()

## cmap과 alpha 옵션
## cmap에 컬러를 지정하면, 컬러 값을 모두 같게 가져갈 수 있다.
## alpha 값은 투명도를 나타내며 0~1의 값. 0에 가까울 수록 투명한 값
np.random.rand(50)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(x, y, s=area, cmap='blue', alpha=0.1)       # 즉 cmap은 컬러를 문자열 형식으로 지정하게 해준다.
plt.title('alpha=0.1')
plt.subplot(1, 3, 2)
plt.scatter(x, y, s=area, cmap='blue', alpha=0.5)
plt.title('alpha=0.5')
plt.subplot(1, 3, 3)
plt.scatter(x, y, s=area, cmap='blue', alpha=1.0)
plt.title('alpha=1.0')

plt.show()

#########################

# Barplot, Barhplot
## bar
x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]

plt.bar(x, y, align = 'center', alpha=0.7, color='red')     # align : ticks이 표시되는 것을 가운데 정렬하라는 의미(center를 줬기 때문)
plt.xticks(x)
plt.ylabel('Number of Students')
plt.title('Subjects')

plt.show()

## barh
x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]

plt.barh(x, y, align = 'center', alpha=0.7, color='green')     # align : ticks이 표시되는 것을 가운데 정렬하라는 의미(center를 줬기 때문)
plt.yticks(x)               # 위와 다르게 yticks라는것을 확인!!
plt.xlabel('Number of Students')
plt.title('Subjects')

plt.show()

# barplot에서 비교그래프 그리기
x_label = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
x = np.arange(len(x_label))
y_1 = [66, 80, 60, 50, 80, 10]
y_2 = [55, 90, 40, 60, 70, 20]

## 넓이 지정
width = 0.35

## subplots 생성
fig, axes = plt.subplots()

## 넓이 설정
axes.bar(x - width/2, y_1, width, align='center', alpha=0.5)
axes.bar(x + width/2, y_2, width, align='center', alpha=0.8)

## xtick 설정
plt.xticks(x)
axes.set_xticklabels(x_label)
plt.ylabel('Number of Students')
plt.title('Subjects')
plt.legend(['john', 'peter'])

plt.show()

# barhplot에서 비교그래프 그리기
x_label = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
x = np.arange(len(x_label))
y_1 = [66, 80, 60, 50, 80, 10]
y_2 = [55, 90, 40, 60, 70, 20]

## 넓이 지정
width = 0.35

## subplots 생성
fig, axes = plt.subplots()

## 넓이 설정
axes.bar(x - width/2, y_1, width, align='center', alpha=0.5, color='green')
axes.bar(x + width/2, y_2, width, align='center', alpha=0.8, color='red')

## xtick 설정
plt.yticks(x)
axes.set_xticklabels(x_label)
plt.xlabel('Number of Students')
plt.title('Subjects')
plt.legend(['john', 'peter'])

plt.show()


# Lineplot
x = np.arange(0, 10, 0.1)
y = 1 + np.sin(x)

plt.plot(x, y)

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin graph', fontsize=18)

plt.grid()
plt.show()

## 2개 이상의 그래프 그리기
x = np.arange(0, 10, 0.1)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)

plt.plot(x, y_1, label='1+sin', color='blue', alpha=0.3, marker='o', linestyle=':')
plt.plot(x, y_2, label='1+cos', color='red', alpha=0.7, marker='+', linestyle='-.')

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin and cos graph', fontsize=18)
plt.legend()
plt.grid()
plt.show()


# Areaplot(filled area)
## matplotlib 에서 area plot을 그리고자 할 때는 fill_between함수를 사용한다.
x = np.arange(1, 21)
y = np.random.randint(low=5, high=10, size=20)

plt.fill_between(x, y, color='green', alpha=0.6)
plt.show()

## 경계선을 굵게 그리고 area는 옅게 그리는 효과 적용
plt.fill_between(x, y, color='green', alpha=0.3)
plt.plot(x, y, color='green', alpha=0.8)        # 이런식으로 영역부분은 좀 더 투명한게 한 뒤 라인그래프를 덮어주는 형식
plt.show()

## 여러 그래프를 겹쳐서 표현
x = np.arange(0, 10, 0.05)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)
y_3 = y_1 * y_2 / np.pi

plt.fill_between(x, y_1, color='green', alpha=0.1)
plt.fill_between(x, y_2, color='blue', alpha=0.2)
plt.fill_between(x, y_3, color='red', alpha=0.3)
plt.show()


# Histogram

N = 1000000
bins = 30       # 나누는 구간 수

x = np.random.randn(N)

plt.hist(x, bins=bins)
plt.show()

## sharey : y축을 다중 그래프가 share(공유)     -> 비교할 때 자주 사용한다.
## tight_layout : graph의 패딩을 자동으로 조절해주어 fit한 graph 생성

fig, axs = plt.subplots(1,3, sharey=True, tight_layout=True)
fig.set_size_inches(12, 5)

axs[0].hist(x, bins=bins)
axs[1].hist(x, bins=bins*2)
axs[2].hist(x, bins=bins*4)

plt.show()

## Y축에 Density 표기(몇퍼센트 몰려 있는지)
fig, axs = plt.subplots(1, 2, tight_layout=True)
fig.set_size_inches(9, 3)
axs[0].hist(x, bins=bins, density=True, cumulative=True)
axs[1].hist(x, bins=bins, density=True)
plt.show()


# Pie chart
## pie chart 옵션
### explode : 파이에서 툭 튀어져 나온 비율
### autopct : 퍼센트 자동으로 표기
### shadow : 그림자 표시
### startangle : 파이를 그리기 시작할 각도

## texts, autotexts 인자를 리턴 받는다.
## texts는 label에 대한 텍스트 효과를
## autotexts는 파이 위에 그려지는 텍스트 효과를 다룰 때 활용한다.

labels = ['Samsung', 'Huawei', 'Apple', 'Xiaomi', 'Oppo', 'Etc']
sizes = [20.4, 15.8, 10.5, 9, 7.6, 36.7]
explode = (0.3, 0, 0, 0, 0, 0)

## texts, autotext인자를 활용하여 텍스트 스타일링을 적용한다.
patches, texts, autotexts = plt.pie(sizes, 
                                    explode=explode,            # 한 조각을 띄워두는 설정
                                    labels=labels, 
                                    autopct='%1.1f%%',          # 소수점 한자릿수로 퍼센트 표시한다는 의미
                                    shadow=True, 
                                    startangle=90)

plt.title('Smartphone pie', fontsize=15)

## label 텍스트에 대한 스타일 적용
for t in texts:
    t.set_fontsize(12)
    t.set_color('gray')

## pie 위의 텍스트에 대한 스타일 적용
for t in autotexts:
    t.set_color('white')
    t.set_fontsize(18)

plt.show()


# Box plot

## 샘플 데이터 생성
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

plt.boxplot(data)
plt.tight_layout()
plt.show()

## 다중 box plot 생성
### 샘플 데이터 생성 코드에 너무 신경쓰지 말고 box plot을 생성하는것을 주요하게 보면 된다.
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d1 = np.concatenate((spread, center, flier_high, flier_low))

spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low))

d1.shape = (-1, 1)
d2.shape = (-1, 1)

d1 = [d1, d2, d2[::2, 0]]

plt.boxplot(d1)
plt.show()

## box plot 축 바꾸기 : vert=False
plt.title('Horizontal box plot', fontsize=15)
plt.boxplot(d1, vert=False)

plt.show()

## outlier 마커 심볼과 컬러 변경 : flierprops
outlier_marker = dict(markerfacecolor='r', marker='D')

plt.title('change outlier symbols', fontsize=15)
plt.boxplot(d1, flierprops=outlier_marker)
plt.show()


# 3D 그래프 그리기
## 3d로 그래프를 그리기 위해서는 mplot3d를 추가로 import 해야한다.
from mpl_toolkits import mplot3d

## 밑그림 그리기(캔버스)
fig = plt.figure()
ax = plt.axes(projection='3d')

## 3d plot 그리기
### project=3d로 설정한다.
ax = plt.axes(projection='3d')

### x, y, z 데이터를 생성한다.
z = np.linspace(0, 15, 1000)
x = np.sin(z)
y = np.cos(z)

ax.plot(x, y, z, 'gray')
plt.show()

### 또 다른 예
ax = plt.axes(projection='3d')

sample_size = 100
x = np.cumsum(np.random.normal(0, 1, sample_size))
y = np.cumsum(np.random.normal(0, 1, sample_size))
z = np.cumsum(np.random.normal(0, 1, sample_size))

ax.plot3D(x, y, z, alpha=0.6, marker='o')

plt.title("ax.plot")
plt.show()

## 3d scatter 그리기
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')      # Axe3D object

sample_size2 = 500

x = np.cumsum(np.random.normal(0, 5, sample_size2))
y = np.cumsum(np.random.normal(0, 5, sample_size2))
z = np.cumsum(np.random.normal(0, 5, sample_size2))

ax.scatter(x, y, z, c=z, s=20, alpha=0.5, cmap='Greens')

plt.title('ax.scatter')
plt.show()

## contour3D 그리기(등고선)
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)

z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')

ax.contour3D(x, y, z, 20, cmap='Reds')

plt.title('ax.contour3D')
plt.show()


# imshow
## 이미지 데이터와 유사하게 행과 열을 가진 2차원 데이터를 시각화 할 때 활용하는 그래프
from sklearn.datasets import load_digits

digits = load_digits()
X0 = digits.images[:10]
X0[0]
print(X0[0])
## load_digits는 0~16 값을 가지는 array로 이루어져 있다.
## 1개의 array는 8X8 배열 안에 표현되어 있다.
## 숫자는 0~9까지 이루어져 있다.

fig, axes = plt.subplot(nrows=2, ncols=5, sharex=True, figsize=(12, 6), sharey=True)

for i in range(10):
    axes[i//5][i%5].imshow(X0[i], cmap='Blues')
    axes[i//5][i%5].set_title(str(i), fontsize=20)

plt.tight_layout()
plt.show()




