# 분석 기법 (피벗 테이블)



```python
%matplolib inline
import pandas as pd

form matplotlib import rc
rc('font', family = 'Malgun Gothic')

마트판매현황 = pd.read_excel('data/합친결과.xlsx')
print(마트판매현황.shape)

마트판매현황.describe()		# 데이터베이스 수치컬럼에 연관된 통계 요약정보를 알려줌

마트판매현황.describe().loc['max']	# 이런식으로 확인 가능
마트판매현황.describe(percentiles=[.10, .20, .30, .40, .50])	#이런식으로 10퍼 지점, 20퍼지점, 30퍼지점 등 확인 가능
마트판매현황['상품명']		# 시리즈 형식
마트판매현황['상품명'].value_counts()		# 현재 시리즈에서 각각의 값들이 몇번 나왔는지 반환
마트판매현황['상품명'].value_counts().sort_values(ascending = False)		# 반환된 정보 내림차순까지
마트판매현황['상품명'].value_counts().sort_values(ascending = False)[:10] 	# 상위 10개까지

마트판매현황.pivot_table(index='상품명')			# 기본적으로 그룹 분석(컬럼 지정 필요), 숫자데이터에서만 한해서 보여준다.
마트판매현황.pivot_table(index='상품명', values = '판매건수')	# values를 통해 보고싶은 값만 볼 수 있다.
마트판매현황.pivot_table(index='상품명')[[판매건수]]		# 이런식으로 데이터프레임 형식으로도 불 수 있다.
마트판매현황.pivot_table(index='상품명', values = ['매출', '판매건수'])	# 리스트로 묶어 여러 항목을 볼 수 있다.
마트판매현황['매출'] = 마트판매현황['가격'] * 마트판매현황['판매건수']
마트판매현황.pivot_table(index='년월', values = ['매출', '판매건수'], aggfunc = 'mean')	# affunc에 함수값을 넣어 진행 가능(max, min, mean, 등)

마트판매현황.pivot_table(index='년월', values = ['매출', '판매건수'], aggfunc = 'sum').plot(kind = 'pie', y = '매출')		# 그래프 그리기가 가능하다.


```

