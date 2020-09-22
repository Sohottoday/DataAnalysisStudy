
import sqlite3

class SqliteDB:
    def __init__(self, dbname):
        print('생성자 출력')
        self.conn = sqlite3.connect(dbname)
        self.cursor = self.conn.cursor()

    def __del__(self):  # 소멸자(마감 처리 용도)
        print('소멸자 출력')
        self.cursor.close()
        self.conn.close()

    def getJoinData(self):  # 조인 데이터
        sql = " select st.id, st.name, st.birth, sg.subject, sg.jumsu"
        sql += " from students st join sungjuk sg "
        sql += " on st.id = sg.id"
        # 문잡을 합칠 때 띄워쓰기를 주의해야 한다.

        result = self.cursor.execute(sql)
        return result

    def getSubQuery(self, name):
        sql = "select * from sungjuk"
        sql += f" where id = (select id from students where name = '{name}')"

        result = self.cursor.execute(sql)
        return result

    def getJumsu(self, name):       # 고아라, 심형래를 구현해보시오.
        sql = "select jumsu from sungjuk"
        sql += f" where id = (select id from students where name = '{name}')"

        result = self.cursor.execute(sql)

        total = 0 # 총점
        cnt = 0 # 행 갯수

        for row in result:
            total += row[0]
            cnt += 1

        if cnt != 0:
            average = total/cnt
            print('총점 : ', total, ', 평균 : ', average)
            return (total, average)
        else:
            print('존재하지 않는 회원입니다.')
            return None

        #if cnt == 0:
#            return None     # 부정
        #average = total/cnt

        #return total, average


dbname = 'sqlitedb.db'
mydb = SqliteDB(dbname)

dataset = mydb.getJoinData()
for row in dataset:
    print(row)
print('#' * 30)

who = '이문세'
dataset = mydb.getSubQuery(who)
for (id, subject, jumsu) in dataset:
    print('아이디 : ', id, end='', sep='')
    print(', 과 목 : ', subject, end='', sep='')
    print(', 점 수 : ', jumsu, end='', sep='')
    print()

print('#' * 30)

namejumsu = '고아라'
dataset = mydb.getJumsu(namejumsu)

#if dataset != None:
#    print('총점 : ', dataset[0], ', 평균 : ', dataset[1])
#else:
#    print('존재하지 않는 회원입니다.')





print('finished')