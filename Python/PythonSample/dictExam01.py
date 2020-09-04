# 사전(dict) : 키와 값으로 구성되어 있는 데이터 구조
# 중괄호를 사용한다
# 키와 값은 콜론으로 구분하고, 항목들은 콤마로 구분한다.
# 예시 : {키1: 값1, 키2:값2, 키3:값3, ...}
# 순서를 따지지 않는 자료구조
# 관련함수 : dict()
mydict = {'name':'홍길동', 'phone':'01011112222', 'birth':'12/25'}
print(mydict)
print(mydict['birth'])

# 읽기 : 사전 이름에 '키'를 넣어주면 '값'을 구할 수 있다.
print(mydict['birth'])

# 쓰기 :
mydict['phone'] = '01033335555'
# 존재하지 않는 키는 insert가 된다.
mydict['address'] = '마포구 공덕동'

# 존재하지 않는 키는 KeyError 오류가 발생한다.
# print(mydict['age'])

# get('찾을 키', 기본값)
print(mydict.get('age', 10))

print(mydict)

del mydict['phone']     # 해당 키를 제거한다.

print(mydict)

if 'address' in mydict:
    print('주소 있음')
else:
    print('주소 없음')

if 'phone' in mydict:
    print('주소 있음')
else:
    print('주소 없음')

mydict.clear()
print(mydict)

mydict = {}     # 비어있는 dict 생성
mydict['hong'] = ['홍길동', 23]
mydict['park'] = ['박영희', 35]
print(mydict)