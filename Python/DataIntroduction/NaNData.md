# 빠진 데이터를 처리하는 방법



- Pandas 내 표현 : numpy.nan -> Not a Number
- NaN 데이터는 대개의 Pandas 연산에서 제외됩니다.



```python
with open("data/~~~.csv", "rt", encoding = "utf8") as f:
    print(f.read())
    
import pandas as pd

df = pd.read_csv('data/~~~.csv', encoding = 'utf8')

df['좋아요'].sum() / 5		# 5개 행의 데이터임
df['좋아요'].mean()		# 위와 아래의 값이 다르다. 이유는 중간에 NaN 값이 존재하므로
# NaN을 0으로 처리하여 진행하려면
df['좋아요'].fillna(0)		# na의 값을 0으로 채우겠다는 의미
df['좋아요'].fillna(0).mean()		# na의 값을 0으로 대체한 후 평균을 구함.

# missingno 라이브러리 : NaN 값을 찾는데 도움을 주는 라이브러리
# 항상 라이브러리를 사용하기 전에는 설치해주자 : pip install missingno

import pandas as pdd
import numpy as np

df = pd.DataFrame(np.random.rand(100, 100))		# 랜덤 숫자로 데이터프레임을 생성, 100행 100열
cond = df > 0.3
df[cond]		# 0.3이 넘는 값 외에는 모두 NaN으로 채워진다.

import missingno as msno
%matplotlib inline		# 시각화를 위해 matplotlib도 설치

msno.matrix(df)		# NaN값은 흰색, 값이 있는곳은 검은색으로 표시

df.isnull()		# NaN 값은 True로 반환	
df.isna()		# NaN 값은 True로 반환
df.notnull()	# NaN 값은 False로 반환
df.notna()		# NaN 값은 False로 반환

df.dropna()		# default 값이 0으로 na값을 0으로 모두 변환
df.dropna('index')	# na 가 존재하는 index 제거
df.dropna('index', how = 'all')		# default값은 any, any는 하나라도 존재하면 index를 제거하고 all은 모두 na값이어야 제거

df.fillna()		# na의 값을 어떠한 값으로 채워넣음
value = df['~~~'].mean()
df['~~~'] = df['~~~'].fillna(value)		# 이러한 방식으로 na값에 평균을 넣어줄수도 있음, 평균 외에 최소값, 최대값도 모두 가능.

# NaN 값은 총 개수에는 포함되지만, 실제 값 계산에선느 제외되는 경우가 많으므로
# 데이터에 NaN 포함여부를 반드시 확인하여 적절히 처리하여 계산 결과에 오류가 없도록 대응해야 한다.

```

