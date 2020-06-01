# 여러 데이터 파일을 하나로 합치기



### 엑셀 파일을 읽는 2가지 방법

- 엑셀 프로그램없이 파이썬 라이브러리만으로 처리
  - pd.read_excel()
  - GUI 엑셀 프로그램을 열 필요없이, 파이썬 단에서 처리
  - 단순히 데이터만 읽어서 합치는 경우에 보다 빠른 처리
- 엑섹 프로그램을 통한 처리
  - xlwings 라이브러리 활용
  - GUI 엑셀 프로그램을 열어야하기에 파일이 많을 경우, 비효율적
  - 엑셀 기능을 활용한 자동화
  - 엑셀파일 포맷 rows 개수 제한 : 65,535(xls) / 2,048,576(xlsx)



```python
import pandas as pd

for month in range(1, 7):
pd.read_excel('~~~.xls', index_col = '상품명')
df['년월'] = '2016-01'

df_list = []
for month in range(1, 7):
    excel_path = 'data/2016-%02d.xls' % month		# 2016- 다음 정수 2자리 넣겠다는 의미
    print(excel_path)
    
    df = pd.read_excel(excel_path, index_col = '상품명')
    df['년월'] = '2016-%02d' % month
    df_list.append(df)

혹은 df_merged = pd.concat(df_list)

df_merged.to_csv('합친결과.csv')
df_merged.to_excel('합친결과.xlsx')		# 합친 결과를 csv 혹은 xlsx 파일로 저장

import pathlib

path11 = pathlib.Path('./data')		# 패스 객체(data라는 디렉토리) 생성
list(path11.iterdir())		# path11라는 객체의 목록을 리스트로 출력

for path in pathlib.Path('./data').iterdir():		#지정한 디렉토리의 모든 파일 리스트 출력
    print(path)

df_list = []
for path in pathlib.Path('./data').glob('2016-*.xls'):		
    print(path)		# glob을 준 뒤 패턴을 입력하여 패턴에 맞는 파일 리스트를 출력할 수도 있다.
    df = pd.read_excel(path, index_col = '상품명')
    df_list.append(df)
len(df_list)

for path in pathlib.Path('./data').glob('**/*.xls'):		# 하위의 여러 디렉토리까지 검색
```



