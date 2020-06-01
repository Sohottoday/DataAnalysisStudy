# 행과 열의 데이터 추가, 변경 및 삭제



```python
import pandas as pd
import xlwings as xw

df.read_excel('data/top100.xlsx', index_col = '곡일련번호')

df['테스트컬럼'] = 10		# 새로운 컬럼 추가, 값들은 10으로 일괄적 부여
df['테스트컬럼'] = range(1, 101, 1)		# 1부터 101 전까지(101 미포함) 1씩 증가하는 값 부여
# 개수가 정확해야함, 정확하지 않을 경우 value error 발생

df.drop(31316695)		# 인덱스가 31316695인 테이블 행 제거
df.drop(31316695, inplace = True)		# 인덱스가 31316695인 테이블 행 제거 후 df 테이블에 즉시 적용
df.drop("커버이미지_주소", axis=1)		# 커버이미지_주소라는 컬럼의 열 전체 제거, axis=0 이 default값이고 행을 의미, axis = 1 또는 axis = "columns"은 지정된 컬럼의 열 제거
df.drop(columns="커버이미지_주소")		# 이런 방식으로 줘도 컬럼명의 열 전체 제거 가능
df = df.drop(columns=["커버이미지_주소", '테스트컬럼'])		# 리스트 형식으로 다중 행 제거 가능

df1 = pd.read_excel('~~~.xls', index_col = '상품명')
df2 = pd.read_excel('~~~2.xls', index_col = '상품명')
df1 + df2 		# 이런식으로 진행하면 각각의 데이터프레임의 매칭되는 것만 더하게 된다.

df3 = df1.append(df2)
df3 = pd.concat([df1, df2])
df3 = pd.concat([df1, df2], sort = True)		# 데이터 추가 후 정렬까지
df3 = pd.concat([df1, df2], axis = 1, sort = True)		# 데이터의 열 추가

```

