# 엑셀 열고 원하는 범위 값 변경하기

### 파이썬 코드에서는 문자열로 경로 구분자를 지정할 때 역슬래시가 아닌 슬래시 "/"를 써도 잘 동작한다.



```python
import pandas as pd
import xlwings as xw

%pwd		## 현재 디렉토리 확인, 파이썬에서는 되지 않고 jupyter notebook에서만 적용됨

workbook = xw.Book('data/2016-01.xls')
workbook
```

```python
workbook.sheets		# 활성화된 시트 확인

workbook.sheets['Sheet1']
workbook.sheets[0]		# 인덱스를통한 접근, 이름을 통한 시트 접근이 모두 가능하다.

sheet = workbook.sheets.active		#활성화된 시트를 변수에 적용
sheet.range('A1').value		# 활성화된 시트의 첫번째 컬럼명 불러오기
data = sheet.range('A1').expand().value		#data라는 변수에 시트를 객체화 시킴
len(data)
pd.DataFrame(data)		#데이트 프레임 형식으로 변경시킴
# 이런식으로 될 경우 컬럼명이 임의의 컬럼명으로 지정 됨. 기존 인덱스는 맨 위 값으로 지정되므로
pd.DataFrame(data[1:]) 	# 이러한 형식으로 컬럼명 변경시켜줌.
# index도 임의의 숫자 인덱스가 아닌 첫 열의 인덱스로 바꿔주고 싶다면
pd.DataFrame(data[1:], columns=columns).set_index('상품명')

df.shape()		# 형식 확인
df.loc['청양고추']		# 키값에 대한 value값들 확인

sheet.range('E2').value = 10		#E2 위치에 10이란 값 지정
sheet.range('E2').value = [10, 20, 30]		#E2부터 가로로 10, 20, 30 값 지정
sheet.range('E2').options(transpose=True).value = [20, 30, 40] 	# E2부터 세로로 값 부여
sheet.range('E8:H12').value = 100		#range 안의 해당 범위에 값 100 부여
sheet.range('E8:H12').clear_contents()		# 해당 범위 안의 내용 비우기(지우기)

매출 = df['판매건수'] * df['가격']		# 판매건수 컬럼과 가격 컬럼의 값 곱해서 매출이라는 컬럼으로 출력해줌
매출.values		# 값 확인 가능
df['매출'] = 매출		# 이런식으로 df라는 데이터프레임에 컬럼 추가도 가능

sheet.range('D1').value = '매출'		# 현재 활성화된 시트의 D1에 매출이라는 컬럼을 추가한 뒤
sheet.range('D2').options(transpose=True).value = 매출.value		# 이런식으로 값을 넣어주는 방법도 가능

workbook.save()		# 저장
workbook.save('c:/sohottoday')		#이런식으로 경로 지정 후 저장도 가능
workbook.app.kill()		# 활성화 되어이쓴 엑셀 창 종료

```

