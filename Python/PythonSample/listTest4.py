# list 자료구조
# 순서를 가지고 있는 데이터 유형
# 인덱싱, 슬라이싱이 가능하고, 대괄호를 사용한다.
# 관련 함수 : list()
# 조작을 위한 함수들 : append()

mylist = ['강감찬', '김유신']
print(mylist)

mylist.append('이순신')
mylist.append('안중근')
print(mylist)
print(len(mylist))
# 마지막 요소 구하기
print(mylist[len(mylist)-1])

mylist.insert(2, '이성계')
print(mylist)       # insert 함수를 활용해 원하는 위치에 삽입이 가능하다.

# mylist.clear()        # 배열을 지움

# 정렬하는 함수 sort
mylist.sort()
print(mylist)

# 배열을 뒤집는 함수 reverse()
mylist.reverse()
print(mylist)

print(mylist[2])
mylist[2] = '이완용'
print(mylist)

# remove() 해당하는 요소를 삭제
mylist.remove('김유신')
print(mylist)

# extend() : 리스트를 병합
newlist = ['윤봉길', '신사임당', '강감찬']
mylist.extend(newlist)
print(mylist)

# count() : 배열 내 해당하는 요소의 개수를 반환
print(mylist.count('강감찬'))

# '이완용'은 몇번째에 있나요?
print('이완용 위치 : ',mylist.index('이완용'))
print('index : ', mylist.index('강감찬', 4))       # 4번째 이후부터 해당하는 요소를 찾아 위치를 반환

# 슬라이싱 테스트
print(mylist[4:6])
print(mylist[1::2])
print(mylist[0::2])

# 요소의 인덱스가 3의 배수인 항목들만 추출
print(mylist[0::3])

# 모든 유형의 데이터 사용이 가능하다.
anydata = [10, '가가', 12.34, [10, 20, 30]]
print(anydata)
