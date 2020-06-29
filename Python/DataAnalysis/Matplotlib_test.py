# 시각화 패키지

# 데이터를 차트(chart)나 플롯(plot)으로 시각화하는 패키지
# 여러 다양한 시각화 기능을 제공한다.

# 라인 플롯(line plot), 스캐터 플롯(scatter plot), 바 차트(bar chart), 히스토그램(histogram) 등

# 서브 패키지
## pylab
## import matplotlib as mpl
## import matplotlib.pylab as plt 로 불러온다.

import matplotlib as mpl
import matplotlib.pylab as plt

plt.plot([1, 5, 8, 15])     # plot 명령은 ndarray 객체를 반환한다.
plt.grid(True)
plt.show()


plt.plot([100, 200, 300, 400], [1, 4, 10, 17], 'rs--')
plt.show()
# 차트의 스타일 지정 순서 : 색상(color), 마커(marker), 선 종류(line style)
# rs-- 의 의미는 앞의 r은 색상(red)를 표현한 것이고 s는 점 모양(square), --는 선 종류를 나타낸 것이다.
# 선 색상은 r(red), m(magenta), c(cyan) 등 이 존재한다.

# Matplotlib가 그리는 그림은 Figure객체, Axes객체, Axis 객체로 구성된다.
# Figure 객체는 한개 이상의 Axes객체를 포함할 수 있다.
# Axes객체는 다시 두개 이상의 Axis 객체를 포함한다.
# 즉, Axis객체는 하나의 플롯(plot)을 의미한다.
# Axis는 세로축(y축)이나 가로축(x축) 등의 축을 의미한다.

# Figure 객체는 Matplotlib.figure.Figure클래스 객체이다
# Figure는 플롯에 그려지는 캔버스(도화지)를 뜻한다.

# subplot : 하나의 Figure 안에 여러개의 플롯(plot)을 배열 형태로 보이도록 할 때 사용한다.
# Figure 안에 Axes를 생성하려면 subplot명령을 사용해서 Axes객체를 얻어야 한다.
# 그러나, plot명령을 사용해도 자동으로 Axes를 생성해 준다.

# subplot(2, 1, 1), subplot(2, 1, 2)
# tight_layout 명령을 실행하면 플롯(Axes)간의 간격을 자동으로 조절해준다.

# np.linespace : Numpy에 존재하는 함수로 (start, stop, num, endpoint=True, restep=False, dtype)
## start(시작값), stop(endpoint가 False로 설정되지 않은 경우 끝 값이 된다.)), num(생성할 샘플 수, 기본값 50, 음수는 될 수 없다),
## endpoint(끝 점), restep(샘플간의 간격을 설정할 수 있는 step을 반환한다.), dtype




