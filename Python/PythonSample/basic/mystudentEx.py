
# mystudent.xml 파일을 읽어서 students.csv 파일을 만들기
# students.db 파일이 students 테이블을 생성하고, 데이터를 추가
# pandas.DataFrame, sqlite, xml
# 추가사항 : csv파일과 테이블에 총점과 평균도 같이 넣어보기

from xml.etree.cElementTree import parse
import pandas as pd
import sqlite3


tree = parse('mystudent.xml')
myroot = tree.getroot()
person = list()


students = myroot.findall('student')
#print(students)

for std in students:

    name = std.find('name').text
    kor = int(std.find('국어').text)
    eng = int(std.find('영어').text)
    math = int(std.find('수학').text)
    total = kor+eng+math
    average = total/3
    person.append((name, kor, eng, math, total, average))


print(person)

mycolumns = ['이름', '국어', '영어', '수학', '총점', '평균']
myframe = pd.DataFrame(person, columns=mycolumns)
myframe.to_csv('students.csv')
print('csv 저장 완료')

conn = sqlite3.connect(database='studentdb.db')
mycursor = conn.cursor()

try:
    mycursor.execute("drop table students")

except sqlite3.OperationalError as err:
    print("테이블이 존재하지 않습니다.")

mycursor.execute(
        '''
        create table students
        (name text, kor integer, eng integer, math integer, total integer, avg integer)
        '''
)

sql = "insert into students(name, kor, eng, math, total, avg) values(?, ?, ?, ?, ?, ?)"
mycursor.executemany(sql, person)
conn.commit()

mycursor.close()
conn.close()
