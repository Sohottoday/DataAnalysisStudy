
# 중첩 리스트
saram01 = ['hong', '홍길동', 20, '용산']
saram02 = ['kim', '김철수', 30, '마포']
saram03 = ['kang', '강남남', 40, '구로']

mylist = []
mylist.append(saram01)
mylist.append(saram02)
mylist.append(saram03)
print(mylist)

print(mylist[1][2])

mylist[2][1] = '강호동'
print(mylist)

# 세 사람의 평균 나이 구하기
print((mylist[0][2]+mylist[1][2]+mylist[2][2])/3)

totalage = mylist[0][2]+mylist[1][2]+mylist[2][2]
print('평균 나이 : %.2f' % (totalage/len(mylist)))

# '홍길동$김철수$강호동' 출력해보기
nameall = []
nameall.append(mylist[0][1])
nameall.append(mylist[1][1])
nameall.append(mylist[2][1])

print('$'.join(nameall))