from pandas import Series
import matplotlib.pyplot as plt

# Series : 1차원적인 동일한 타입의 데이터 묶음
# 엑셀 파일의 특정한 1개의 컬럼 정보

# 색인을 지정하지 않으면 0부터 순차적으로 숫자를 매긴다.
mylist = [10, 40, 30, 20]
myseries = Series(mylist)
print(myseries)
print('-' * 30)

# 색인을 지정하려면 index 매개 변수를 사용한다.
myseries = Series(mylist, index=['강감찬', '이순신', '김유신', '광해군'])
# 색인에 이름 지정하기
myseries.index.name = '정수'

# 시리즈 자체에 이름 지정하기
myseries.name = '학생들 시험'

print(myseries)
print('-' * 30)

print(type(myseries))
print('-' * 30)

# 문자열의 dtype은 'object'입니다.
print(myseries.index)
print('-' * 30)

print(myseries.values)      # 색인의 값을 출력
print('-' * 30)

for idx in myseries.index:
    print('색인 : ' + idx + ', 값 : ' + str(myseries[idx]))

print('-' * 30)

myindex1 = ['서울', '부산', '광주', '대구', '울산', '목포', '여수']
mylist1 = [50, 60, 40, 80, 70, 30, 20]
myseries1 = Series(data=mylist1, index=myindex1)
print(myseries1)
print('-' * 30)

print(myseries1['대구'])          # 타입 확인을 잘 해야 한다.
print(type(myseries1['대구']))
print('-' * 30)

print(myseries1[['대구']])        # 대괄호 2개를 통해 시리즈타입인것을 알 수 있다.
print(type(myseries1[['대구']]))
print('-' * 30)

# 문자열 색인으로 슬라이싱 하는 경우 양쪽 모두 포함된다.
print(myseries1['대구':'목포'])
print('-' * 30)

print(myseries1[[2]])
print('-' * 30)

# 콜론으로 슬라이싱 하는 경우에는 대괄호 1개
print(myseries1[0:5:2])
print('-' * 30)

# 콤마를 사용하는 경우 대괄호 2개
print(myseries1[[1, 3, 5]])
print('-' * 30)

myseries1[2] = 22       # 쓰기
print(myseries1)
print('-' * 30)

myseries1[2:5] = 22
print(myseries1)
print('-' * 30)

# 서울과 대구만 55로 변경하기
myseries1[['서울', '대구']] = 55
print(myseries1)
print('-' * 30)

myseries1[0::2] = 77
print(myseries1)
print('-' * 30)

plt.rc('font', family='Malgun Gothic')
myseries1.plot(kind='bar', rot=0)
plt.savefig('graph05.png')
