# 차트를 통해 데이터를 한 눈에 이해하기





```python
%matplotlib inline		# 그래프 등을 보여줄 수 있게 설정하는 코드
import pandas as pad
import xlwings as xw

# 만약 한글이 짤려나온다면 한글을 지원하지 않는 폰트여서 그런 경우도 존재
from matplotlib import rc
rc('font', family = 'Malgun Gothic')		# 메타플롯립을 임포트하여 맑은 고딕체로 지정해줌.
#rc('font', family = 'AppleGothic')		# iMac일 경우 애플고딕으로 설정.

from matplotlib import font_manager
font_manager.fontManager.ttflist


df = pd.read_excel('~~~.xls')
print(df.shape)		# 데이터의 형태 확인

판매건수_top10 = df.sort_values('판매건수', ascending = False).head(10)		# 판매건수 내림차순 상위 10개 출력

판매건수_top10.plot()		# 한매건수_top10이란 변수의 그래프 출력, 기본으론 꺽은선 그래프
판매건수_top10.plot(kind='bar')		# 막대그래프, 원형그래프는 kind에 'pie' 입력
판매건수_top10[['판매건수', '가격']].plot(kind='bar')		# 이런식으로 보고싶은 항목만 그래프로 출력 가능
판매건수_top10[['판매건수', '가격']].plot(kind='barh')		# 가로 막대형 그래프
판매건수_top10[['판매건수', '가격']].sort_values('가격', ascending = False).plot(kind='barh') 	# 응용
# pie 그래프는 한개의 주제에 대해서만 시각화 가능
판매건수_top10[['판매건수', '가격']].sort_values('가격', ascending = False).plot(kind='pie', y='가격') 		# 가격에 대한 파이그래프
판매건수_top10[['판매건수', '가격']].sort_values('가격', ascending = False).plot(kind='pie', y='판매건수') 		# 판매건수에 대한 파이그래프

# 그래프를 엑셀 파일에 집어넣기
xw.view(판매건수_top10)		# 데이터 객체에 대한 엑셀창 열기
ax = 판매건수_top10[['판매건수', '가격']].sort_values('가격', ascending = False).plot(kind='bar')		# 데이터에 대한 그래프 생성하여 객체에 담음
fig = ax.get_figure()		# 생성된 그래프를 피규어 객체로 만듬
sheetaaa = xw.sheets.active		# 현재 활성화된 워크시트를 객체로 담음
leftaa = sheetaaa.range('F1').left
topaa = sheetaaa.range('F1').top

sheetaaa.pictures.add(fig, name = "2016년 1월 판매량 TOP 10", update = True, left = leftaa, top = topaa, grid = True, figsize = (8,3), rot=0)		# 활성화된 워크시트를 담은 객체를 통해 그래프 추가. //  name을 통해 차트끼리 구별하기 위한 이름 지정, update를 true로 주면 데이터가 변경될 경우 그래프도 변경됨, left와 top에 차트가 들어갈 위치 지정
# grid = True 를 주면 그래프 배경에 줄 표현
# figsize = (9,3) 을 주면 그래프의 사이즈 지정
# rot = 0 을 통한 정렬?
# 이미지 사이즈 조정은 이전 이미지를 삭제하고 해야 사이즈 조정이 간편하다(이미지 찌그러짐 현상 방지)

# 엑셀이 주기적으로 업데이트되는 코드를 만들어 진행한다면 매번 엑셀 보고서를 작성해야 하는 불편함이 방지됨.

```

