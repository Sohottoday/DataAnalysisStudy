# 엑셀 파일 분석
# - 파이썬에서 엑셀 파일을 분석하기 위해서는 파이썬 엑셀 라이브러리를 설치해야 한다.
# -> pip install openpyxl
# 엑셀 파일의 구조
# 보통 엑셀 파일을 book이라 부른다. book 내부에는 여러개의 sheet가 존재한다.
# 각 시트는 행(row)과 열(column)을 가진 2차원의 셀(cell)로 구성되어 있다.

import openpyxl
import xlrd

# 엑셀 파일 불러오기 : load_workbook(파일명)
fileName = "stats_106001.xls"
book = openpyxl.load_workbook(fileName)
#book = xlrd.open_workbook(fileName)


# 엑셀 파일에서 원하는 sheet를 추출하기
# worksheets[인덱스] : 인덱스가 0, 1, 2~
sheet = book.worksheets[0]      # 첫 번째 시트를 가져온다는 의미

# 시트으 각 행을 순서대로 추출해보기
excel_data = []

for row in sheet.rows:
    excel_data.append([
        row[2].value,
        row[9].value
    ])

# 필요없는 행(헤더)제거하기
del excel_data[0]

for data in excel_data:
    print(data[0], data)

