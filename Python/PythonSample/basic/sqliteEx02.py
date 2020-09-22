
import sqlite3

conn = sqlite3.connect(database='sqlitedb.db')

mycursor = conn.cursor()

try:
    mycursor.execute("drop table sungjuk")
except sqlite3.OperationalError as err:
    print("테이블이 존재하지 않습니다.")

mycursor.execute(
        '''
        create table sungjuk
        (id text, subject text, jumsu integer)
        '''
)

#sql = "insert into sungjuk(id, subject, jumsu) values('lee', 'java', '10')"
#mycursor.execute(sql)

datalist = [('lee', 'java', 10), ('lee', 'html', 20), ('lee', 'python', 30), ('jo', 'java', 40), ('jo', 'html', 50), ('jo', 'python', 60), ('ko', 'java', 70), ('ko', 'html', 80), ('ko', 'html', 90)]
sql = "insert into sungjuk(id, subject, jumsu) values(?, ?, ?)"
mycursor.executemany(sql, datalist)
conn.commit()

mycursor.close()
conn.close()

print('finished')


