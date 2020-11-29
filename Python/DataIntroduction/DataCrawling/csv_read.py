# # csv 데이터 분석
# - 웹에서 가장 많이 사용되는 형식
# - 엑셀로 쉽게 만들 수 있는 형식
# - 수많은 데이터베이스와 데이터 도구에서 csv 형식을 지원

# csv(comma-separated values) 파일은 텍스트 에디터를 이용해서 간편하게 수정할 수 있다.

# csv와 비슷한 형식으 파일들로 tsv(tab-separated values) : 콤마가 아닌 탭으로 필드를 구분하는 형식
# ssv(space-separated values) : 공백으로 필드를 구분하는 형식

import codecs

fileName = "prod_list.csv"
csv = codecs.open(fileName, "r", "euc_kr").read()

data = []
records = csv.split("\r\n") # \r:CR \n:LF(new line)

for rec in records:
    if rec == "":
        continue
    fields = rec.split(",")
    data.append(fields)

for field in data:
    print(field[1], field[2])

# 파이썬 csv 모듈을 이용한 csv 데이터 처리 방법
# csv 파일에 있는 필드 데이터가 큰따옴표("")로 둘러 쌓인 경우에는 csv파일을 분석하기 어렵다.
# 따라서, 이 때에는 csv 모듈을 이용하는데 csv.reader(파일명, delimiter=",", quotechar='"')

# 여기서 delimiter는 구분문자를 지정하고, quotechar는 어떤 기호로 데이터를 감싸고 있는지를 지정한다.
# csv.writer(파일포인터, delimiter=",", quotechar='"')

import csv

with codecs.open("test.csv", "w", "euc_kr") as fp:
    writer = csv.writer(fp, delimiter=",", quotechar='"')
    writer.writerow(["상품코드", "상품이름", "가격"])
    writer.writerow("1", "키보드", 20000)
    writer.writerow("2", "마우스", 10000)
    writer.writerow("3", "모니터", 100000)


fileName = "test.csv"
ffp = codecs.open(fileName, "r", "euc_kr")

reader = csv.reader(ffp, delimiter=",", quotechar='"')

for field in reader:
    print(field[1], field[2])

