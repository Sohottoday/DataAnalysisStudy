# Overview - 시각화



- 정적 이미지 생성 라이브러리
  - Matplotlib, Seaborn 등
- Interactive 시각화 라이브러리
  - Bokeh, d3py 등



- anaconda를 통해 기본 설치되는 라이브러리
  - pandas : 데이터 전처리 / 분석 라이브러리
  - matplotlib / seaborn : 시각화 라이브러리
  - xlwings : 엑셀 UI 자동화 라이브러리
  - scikit-learn : 머신러닝 라이브러리
  - requests : HTTP  요청 라이브러리
  - beautifulsoup4 : HTML parser 라이브러리
  - tqdm : 진행상태 표시 라이브러리



- 셀 > conda create -n 이름 anaconda		#아나콘다 라이브러리 설치
- 쉘 > activate 이름



### 초기화 코드

```python
%matplotlib inline

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

from tqdm import tqdm_notebook
from libs import stock_daum, stock_naver

# 오류가 뜬다면 python: pip/anaconda: conda		에서 필요한 라이브러리 설치
```

```python
x = [0, 1, 2, 3]
y = [1, 4, 9, 16]

plt.plot(x, y)
plt.title("Line Plot")		# 차트 제목 설정
plt.xlabel("x축")		# x축 라벨이름 설정
plt.ylabel("y축")
#plt.show()		만약 바로 보여지지 않는다면 plt.show() 코드를 실행시켜 보자	

```

