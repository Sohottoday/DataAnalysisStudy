import openpyxl

fileName = "people_data.xlsx"

book = openpyxl.load_workbook(fileName)

# 첫 번째 시트 추출
sheet = book.worksheets[0]

# 시트의 행을 순서대로 추출
excel_record = []

for record in sheet.rows:
    excel_record.append([
        record[0].value,
        record[10].value
    ])

del excel_record[0]
del excel_record[0]
del excel_record[0]
del excel_record[0]
# del excel_record[21]
# del excel_record[22]

# for data in excel_record:
#     data[1] = data[1].replace(',','')
#     data[1] = int(data[1])

# 데이터를 인구순으로 정렬
excel_record = sorted(excel_record, key=lambda x: x[1])

# for data in excel_record:
#     print(data)

# 하위 5개 지역 추출
# enumerate 메서드는 순서가 있는 자료형을 입력 받아 인덱스값을 포함하는 객체를 리턴하는 메서드

for i, name in enumerate(excel_record):
    
    print(i+1, name[0], name[1])


# book.active : 활성화되있는 워크시트를 가져온다.
sheet2 = book.worksheets[0]

# 서울과 계에 해당하는 값을 제외한 인구수 합산하기
for i in range(0, 10):
    totalValue = sheet2[str(chr(i+66))+"4"].value.replace(',', '')
    total = int(totalValue)
    seoulValue = sheet2[str(chr(i+66))+"5"].value.replace(',', '')
    seoul = int(seoulValue)

    result = format(total - seoul, ',')
    print("서울을 제외한 인구 : ", result)

# 엑셀 파일에 결과를 쓰기




