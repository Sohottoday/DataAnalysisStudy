import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'


# scatter
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

# barplot, barhplot
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


# lineplot
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
