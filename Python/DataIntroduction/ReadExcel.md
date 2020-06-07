# 파이썬 엑셀 라이브러리

## openpyx

- https://openpyxl.readthedocs.io
- 엑셀 2010파일 read/write 지원, pandas의 read_excel 함수에서 내부적으로 사용



## pyexcel

- https://github.com/pyexcel/pyexcel
- csv, ods, xls, xlsx, xlsm 등의 파일들에 대해 하나의 API로 접근 지원



## xlswriter

- https://xlswriter.readthedocs.io/
- xlsx 포맷에 대한 생성을 풍성하게 지원



## xlrd

- https://github.com/python-excel/xlrd



## xlwt

- https://github.com/python-excel/xlwt
- 엑셀 97/2000/XP/2003 포맷에 대한 읽기/쓰기를 지원



## xlwings

- https://www.xlwings.org/
- 엑셀 프로그램 자동화 라이브러리, 유일하게 엑셀 프로그램에 의존적
  - 엑셀의 익숙함과 파이썬의 강력함의 콜라보
  - 다른 엑셀 라이브러리들은 엑셀 프로그램과의 연동이 아니라, 엑셀 파일 포맷을 지원하는 형태
- 엑셀을 띄워놓고, 파이썬을 통한 값 가져오기/변경 지원
- 엑셀의 매크로 기능을 파이썬으로 구현 지원
  - VBA 대체 기능
- 설치 : Anaconda Python에 포함
  - 쉘>pip install xlwings 혹은 conda install xlwings
- 윈도우/맥 지원

```python
import xlwings as xw

xw.__version__		## 버전 확인

data = [
    ['apple', 'banana', 'berry'],
    [100, 230, 90],
    [140, 150, 40]
]		# 2차원 데이터 정의(데이터프레임)
data

xw.view(data)		# 엑셀을 연동하여 데이터프레임 형식을 보여줌
```

```python
import numpy as np
data = np.random.rand(3, 5)
data		# 랜덤한 값이 담김
xw.view(data) 	#새로운 워크북으로 오픈
xw.sheets.active	# 키려는 주소를 알게 해 줌
xw.view(data, xw.sheets.active)		# 기존에 켜 있는 워크북에 오픈
```

```python
import pandas as pd
url = 'https://finance.naver.com/marketindex/exchangeList.nhn'
df_list = pd.read_html(url)
len(df_list)

df = df_list[0]
print(df.shape)
df.head()

# 정의되지 않은 컬럼 명들을 지정해 줄 때
df.columns = [
    '통화명',
    '매매기준율',
    '현찰 - 사실 때',
    '현찰 - 파실 때',
    '송금 - 보내실 때',
    '송금 - 받으실 때',
    '인화환산율'
]		
df.head()

# index 변경
df = df.set_index('통화명')	# 기존에 임의로 설정되있던 인덱스인 1,2,3,4,5 의 숫자 없어짐
df.head()
xw.view(df, xw.sheets.active)

# 범위 지정
xw.Range('A1')
xw.Range('A1:C2')
xw.Range('A1:D7').value		# 값 읽어오기
xw.Range('좌상단').expend()		# 자동으로 끝까지 읽어옴, xw.Range('좌상단').expend('table') // expend 값은 'table' 이 dafault값
xw.Range('좌상단').expend('right')	# 좌상단부터 오른쪽끝까지만 지정됨
# expend => table, right, down 등의 값 지정 가능.

data = xw.Range('A1:A3').expend().value
data
```





