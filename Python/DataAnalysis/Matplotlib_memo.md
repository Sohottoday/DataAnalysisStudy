# Matplotlib

- 시각화 패키지

- 데이터를 차트(chart)나 플롯(plot)으로 시각화하는 패키지

  여러 다양한 시각화 기능을 제공한다.

- 라인 플롯(line plot), 스캐터 플롯(scatter plot), 바 차트(bar chart), 히스토그램(histogram) 등

- 서브 패키지

  - pylab

    import matplotlib as mpl

    import matplotlib.pylab as plt 로 불러온다.

``` python
import matplotlib as mpl
import matplotlib.pylab as plt

plt.plot([1, 5, 8, 15])		# plot 명령은 ndarray 객체를 반환한다.
plt.grid(True)
plt.show()
```

![캡처1](https://user-images.githubusercontent.com/58559786/86025333-08314300-ba69-11ea-965e-9ef3557570ec.PNG)

``` python
plt.plot([100, 200, 3030, 400], [1, 4, 10, 17], 'rs--')
plt.show()
```

![캡처2](https://user-images.githubusercontent.com/58559786/86025451-2ac35c00-ba69-11ea-89ad-c0609ee3d2a8.PNG)



- 차트의 스타일 지정 순서 : 색상(color), 마커(marker), 선 종류(line style)

  - `rs--`의 의미는 앞의 r은 색상(red)를 표현한 것이고 s는 점 모양(square),  --는 선 종류를 나타낸 것이다.
  - 선 색상은 r(red), m(magenta), c(cyan) 등 이 존재한다.

  

- Matplotlib가 그리는 그림은 Figure객체, Axes객체, Axis객체로 구성된다.

  Figure 객체는 한개 이상의 Axes객체를 포함할 수 있다.

  Axes객체는 다시 두개 이상의 Axis 객체를 포함한다.

  즉, Axis 객체는 하나의 플롯(plot)을 의미한다.

  Axis는 세로축(y축)이나 가로축(x축) 등의 축을 의미한다.

  

- Figure 객체는 Matplotlib.figure.Figure클래스 객체이다.

  Figure는 플롯에 그려지는 캔버스(도화지)를 뜻한다.



- subplot : 하나의 Figure 안에 여러개의 플롯(plot)을 배열 형태로 보이도록 할 때 사용한다.

  Figure 안에 Axes를 생성하려면 subplot 명령을 사용해서 Axes 객체를 얻어야 한다.

  그러나, plot명령을 사용해도 자동으로 Axes를 생성해 준다.

  subplot(2, 1, 1), subplot(2, 1, 2)

- tight_layout 명령을 실행하면 플롯(Axes)간의 간격을 자동으로 조절해준다.



- np.linspace : Numpy에 존재하는 함수로 (start, stop, num, endpoint=True, retstep=False, dtype)
  - start(시작값)
  - stop(endpoint가 False로 설정되지 않은 경우 끝 값이 된다.)
  - num(생성할 샘플 수, 기본값 50, 음수는 될 수 없다.)
  - endpoint(끝 점), restep(샘플간의 간격을 설정할 수 있는 step을 반환한다)

``` python
print(np.linspace(2.0, 3.0, num=5))
# [2.   2.25 2.5  2.75 3.  ]

print(np.linspace(2.0, 3.0, num=5, endpoint=False))
# [2.  2.2 2.4 2.6 2.8]

print(np.linspace(2.0, 3.0, num=5, retstep=True))
# (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)			# 2개의 변수에 따로 담을 수 있다.
```



- 비교해보기
- xlim : x축의 범위를 지정한다.
- ylim : y축의 범위를 지정한다.

``` python
N = 8
y = np.zeros(N)
print(N)
# [0. 0. 0. 0. 0. 0. 0. 0.]

x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)

plt.plot(x1, y, 'o')		# 칼라 없이 마커만 o로 출력된다.
plt.plot(x2, y+0.4, 'o')

plt.ylim([-0.5, 1])		# y축의 값이 -0.5에서 1 사이값으로 축을 지정한다.
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/58559786/86164780-4acc4b80-bb4d-11ea-9cfe-7717591a60b8.png)



``` python
X = np.linspace(-np.pi, np.pi, 256)
C = np.cos(X)
plt.plot(X, C)
plt,xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1, 0, 1])
plt.grid(True)
plt.show()
```

![Figure_2](https://user-images.githubusercontent.com/58559786/86164977-9979e580-bb4d-11ea-9bf6-5d5137bfbee0.png)

- tick : plot이나 chart에서 축상의 위치 표시 지점을 의미한다.
  - tick에 씌어지는 숫자나 글자를 틱 라벨(tick label)이라 한다.
  - 일반적으로 tick label은 Matplotlib가 자동으로 정해준다.
  - 사용자가 수동으로 설정을 하고싶다면 xticks, yticks 명령을 사용하여 x축과 y축 설정이 가능하다.
  - 틱라벨 문자열을 수학 기호로 표시하고 싶은 경우 $$ 사이에 LaTeXㅅ 수학문자식을 넣어서 사용한다.

``` python
X = np.linspace(-np.pi, np.pi, 256)
C = np.cos(X)
plt.plot(X, C)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
plt.yticks([-1, 0, 1], ['Low', '0', 'High'])
plt.show()
```

![Figure_3](https://user-images.githubusercontent.com/58559786/86165173-e8c01600-bb4d-11ea-8a9b-f62b373192cb.png)



``` python
xx1 = np.linspace(0.0, 5.0)
xx2 = np.linspace(0.0, 2.0)

yy1 = np.cos(2 * np.pi * xx1) * np.exp(-xx1)
yy2 = np.cos(2 * np.pi * xx2)

ax1 = plt.subplot(2, 1, 1)
plt.plot(xx1, yy1, 'ro-')
ax2 = plt.subplot(2, 1, 2)
plt.plot(xx2, yy2, 'b.-')

plt.show()
```

![Figure_4](https://user-images.githubusercontent.com/58559786/86165272-0ab99880-bb4e-11ea-88f6-cbae30d52bcd.png)



## 다양한 차트

### bar chart

- `bar(x, y)` : x는 x축의 위치, y축의 값

``` python
y = [2, 1, 3]
x = np.arange(3)
xlabel = ['A', 'B', 'C']
plt.bar(x, y)
plt.xticks(x, xlabel)
plt.show()
```

![Figure_5](https://user-images.githubusercontent.com/58559786/86281735-eecef900-bc18-11ea-8654-07f8b201ead0.png)

``` python
np.random.seed(0)
yLabel = ['A', 'B', 'C', 'D']
yPos = np.arange(4)
yValue = 2+10*np.random.rand(4)

plt.barh(yPos, yValue, alpha=0.5)		# alpha는 투명도를 의미한다. 0 ~ 1
plt.yticks(yPos, yLabel)
plt.show()
```

![Figure_6](https://user-images.githubusercontent.com/58559786/86281894-35245800-bc19-11ea-8758-38c9994fbec4.png)



### historam

``` python
x = np.random.randn(1000)
bins = plt.hist(x, bins=10)
print(bins)
#(array([  9.,  20.,  70., 146., 218., 239., 161.,  86.,  37.,  14.]), array([-3.04614305, -2.46559324, -1.88504342, -1.3044936,  #-0.72394379,
#       -0.14339397,  0.43715585,  1.01770566,  1.59825548,  2.1788053 ,
#        2.75935511]), <a list of 10 Patch objects>)

arrays, bins, patchs = plt.hist(x, bins=10)
print(bins)
#[-3.04614305 -2.46559324 -1.88504342 -1.3044936  -0.72394379 -0.14339397
#  0.43715585  1.01770566  1.59825548  2.1788053   2.75935511]

plt.hist(x, bins=10)
plt.show()
```

![Figure_7](https://user-images.githubusercontent.com/58559786/86282140-9cdaa300-bc19-11ea-942f-a5b0a2378443.png)



### pie chart

- 원의 형태를 유지하기 위해서 plt.axis('equal') 명령을 실행한 후 그린다.
  - ratio : 비율
  - explode : 강조하려는 부분
  - labels
  - color : 색
  - autopct : 퍼센트를 나타낼 때 소수점 자리수를 설정해주는 속성
  - shadow : 그림자를 넣어 입체감 있게 표현할 것인지
  - startangle : 시작하는 각도, 반시계 반향으로 그려진다.

``` python
labels = 'A', 'B', 'C', 'D'
ratio = [10, 30, 40, 20]
colors = ['red', 'blue', 'yellowgreen', 'pink']
explode = (0, 0.1, 0, 0)

plt.pie(ratio, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/58559786/86371271-a74d7900-bcbb-11ea-9706-d46b797e71f2.png)

### scatter plot

- 산점도
- 두개의 실수 데이터 집합의 상관관계를 살펴볼 때 많이 사용하는 차트

``` python
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.scatter(X, Y)
plt.show()
```

![Figure_3](https://user-images.githubusercontent.com/58559786/86371741-3d819f00-bcbc-11ea-98f5-b6d6cfdf3072.png)

- np.random.normal : 평균과 표준편차를 활용한 랜덤 값 추출
  - numpy의 np.random.normal(평균, 표준편차, 샘플 수)

``` python
m, sigma = 0, 0.1
s = np.random.normal(m, sigma, 1000)
plt.hist(s, 30)
plt.show()
```

![Figure_2](https://user-images.githubusercontent.com/58559786/86371535-f1cef580-bcbb-11ea-98cf-d0c9c7078e8a.png)

- 데이터가 2차원이 아닌 3차원 이상인 경우에는 점 하나의 크기 또는 칼라를 이용하여 표현한다.

  이러한 차트를 bubble chart라 한다.

  - 크기를 표현할 때 s, 칼라는 c 인수를 사용한다.

``` python
n = 30
x = np.random.rand(n)
y1 = np.random.rand(n)
y2 = np.random.rand(n)
y3 = np.pi * (15 * np.random.rand(n)) ** 2
plt.scatter(x, y1, c=y2, s=y3)
plt.grid(True)
plt.show()
```

![Figure_4](https://user-images.githubusercontent.com/58559786/86372074-af59e880-bcbc-11ea-8611-05136dabc5a7.png)

### triangular grid

- 삼각 그리드 지원을 해주는 패키지
- `import matplotlib.tri as mtri`
- 삼각 그리드는 Triangulation 클래스를 이용하여 생성한다.
  - Triangulation(x, y, triangles) -> 3가지 매개변수를 받는다.
  - triangles는 생략했을 경우 자동으로 생성된다.

``` python
import matplotlib.tri as mtri

x = np.array([0, 1, 2])
y = np.array([0, np.sqrt(3), 0])
triangles = ([0, 1, 2])
triang = mtri.Triangulation(x, y, triangles)
plt.triplot(triang, 'ro-')
plt.xlim(-0.1, 2.1)
plt.ylim(-0.1, 1.8)
plt.show()
```

![Figure_5](https://user-images.githubusercontent.com/58559786/86372503-35762f00-bcbd-11ea-8aba-bec4d1d9ee17.png)

``` python
x = np.array([0, 1, 2, 3, 4, 2])
y = np.array([0, np.sqrt(3), 0, np.sqrt(3), 0, 2*np.sqrt(3)])
triangles = [[0, 1, 2], [2, 3, 4], [1, 2, 3], [1, 3, 5]]
triang = mtri.Triangulation(x, y, triangles)
plt.triplot(triang, 'bo-')
plt.grid()
plt.show()
```

![Figure_6](https://user-images.githubusercontent.com/58559786/86372684-7d955180-bcbd-11ea-89ed-e305d79e126f.png)

