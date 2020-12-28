import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('house_price_clean.csv')

# 만약 colab 사용할 때 한글 폰트가 깨지는 현상이 나타난다면

"""
!apt -qq -y install fonts-nanum > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=10)
fm._rebuild()

# 그래프에 retina display 적용
%config InlineBackend.figure_format = 'retina'

# Colab의 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic')

# 상단 메뉴 - 런타임 - 런타임 다시 시작을 클릭

# 위 코드를 한번 더 실행
!apt -qq -y install fonts-nanum > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=10)
fm._rebuild()

# 그래프에 retina display 적용
%config InlineBackend.figure_format = 'retina'

# Colab의 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic')

# 필요한 패키지, 데이터 로딩

import pandas as pd

df = pd.read_csv('house_price_clean.csv')
df.plot()

# 만약 그래프가 너무 작게 보인다면?
plt.rcParams["figure.figsize"] = (12, 9)
"""

df['분양가'].plot()

# kind 옵션
## line : 선 그래프
## bar : 바 그래프
## barth : 수평 바 그래프
## hist : 히스토그램
## kde : 커널 밀도 그래프
## hexbin : 고밀도 산점도 그래프
## box : 박스 플롯
## area : 면적 그래프
## pie : 파이 그래프
## scatter : 산점도 그래프

df_seoul = df.loc[df['지역']=='서울']      # 서울 지역만 보기 위해 설정
df_seoul_year = df_seoul.groupby('연도')['분양가'].mean()

df_seoul_year.plot(kind='line')     # kind 의 옵션만 바꿔주면 다양한 그래프를 출력 가능하다.

## 히스토그램은 분포-빈도를 시각화하여 보여준다.(가로축에는 분포, 세로축에는 빈도)
## 커널 밀도 그래프 : 히스토그램과 유사하게 밀도를 보여주는 그래프, 히스토그램보다 조금 더 부드러운 라인을 가지고 있다.
df_seoul_year.plot(kind='kde')
## Hexbin : 고밀도 산점도 그래프, x와 y키 값을 넣어주어야 한다. x, y값 모두 numeric한 값을 넣어야 한다.
## 데이터의 밀도를 추정한다.
df.plot(kind='hexbin', x='분양가', y='연도', gridsize=20)

# max값과 아웃라이어의 값의 의미는?
"""
IQR(Inter Quantile Range)
max값은 데이터의 가장 큰 값이 아닌 3Q값에서 1Q값을 뺀 뒤 1.5를 곱해 해당 값을 3Q 값에 더해준 값이다.
ex) IQR= (7732 - 6519.75) * 1.5 = 1818.375
박스플롯 max = 7732 + IQR = 9550.375
박스플롯 min = 6519.75 - IQR = 4701.375

이러한 과정은 outlier을 감지 할 때 가장 많이 활용된다.
"""

## area 그래프는 line 그래프에 단순히 색칠이 되있는 그래프
df.groupby('연도')['분양가'].count().plot(kind='pie')
## scatter plot(산점도 그래프) : 점으로 데이터를 표기, x, y값을 넣어줘야 한다. x축과 y축을 지정해주면 그에 맞는 데이터 분포도를 볼 수 있다.
df.plot(x='월', y='분양가', kind='scatter')
