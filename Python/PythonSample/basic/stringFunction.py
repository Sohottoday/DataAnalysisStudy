# 문자열 관련 함수 다뤄보기

mystring = 'Hello python!'

print('길이 : ', len(mystring))
print('포함 갯수 : ', mystring.count('o'))
print('검색 위치 : ', mystring.find('e'))
print('검색 위치2 : ', mystring.find('o', 6))
print('검색 위치3 : ', mystring.rfind('o'))

print('문자열 치환1 : ', mystring.replace('l', 't'))
print('문자열 치환2 : ', mystring.replace('l', 't', 1))
print('문자열 치환3 : ', mystring.replace('l', 'blue'))

# print의 sep 옵션은 기본 값이 띄워 쓰기
# 띄워 쓰기를 하지 않으려고 sep=''을 사용
# strip() 함수는 기본 값으로 공백을 제거하는데, 임의의 문자를 내가 지정할 수 있다.
mystr = '    가나  다라      '
print('공백 제거 1 : [', mystr.strip(), ']' , sep='')
print('공백 제거 2 : [', mystr.lstrip(), ']')
print('공백 제거 3 : [', mystr.rstrip(), ']')

mystring = 'xxxhello'
print('공백 제거 4 : ', mystring.strip('x'))

mystring = 'hello python'
print('대문자 : ', mystring.upper())
print('소문자 : ', mystring.lower())
print('첫 글자만 대문자 : ', mystring.capitalize())

# delimiter : 문자열을 나눠주는 구분자
# split 함수는 기본 값으로 공백을 이용하여 문자를 분리해준다.
print('문자열 분리 1 : ', mystring.split())

mystring = '소녀시대/빅뱅/원더걸스'
# split 함수는 사용자가 문자열을 지정하면 지정한 문자를 이용하여 분리해준다.
print('문자열 분리 2 : ', mystring.split('/'))

mystring = 'hello_python.xls'

# startswith('H') : H로 시작하는가? (대소문자도 구분함)
print('시작 여부 : ', mystring.startswith('H'))
print('종료 여부 : ', mystring.endswith('.ppt'))

# 메서드 체이닝 : 메서드를 2개 이상 결합하는 것
# 대소문자 구분하지 않고, h로 시작하는지?
print('시작 여부 : ', mystring.upper().startswith('H'))         # 이와 같은식으로 2가지 이상 함수 결합 가능

mylist = ['삼성', '엘지', 'sk']
print('#'.join(mylist))

str = input('문자 입력 : ')    # Korea
pos = 2
# 인덱싱 : 인덱스를 이용하여 특정 부위의 요소를 1개 추출해 내는 것
munja = str[pos]
print(munja)

# is 로 시작하는 함수들은 참 또는 거짓을 반환해준다.
print('대문자 여부 : ', munja.isupper())
print('소문자 여부 : ', munja.islower())
print('숫자 여부 : ', munja.isdigit())

# 프로그램 내부에서 아스키 코드로 변경이 된 다음, 비교 연산이 이루어진다.
print(munja > 's')
print(munja >= 'A' and munja <= 'Z')         # 대문자인지 물어보는 것
print(munja >= 'a' and munja <= 'z')         # 소문자인지 물어보는 것
print(munja >= '0' and munja <= '9')         # 숫자인지 물어보는 것

# ord 함수는 문자를 아스키 코드로 바꿔주는 함수이다.
print(ord(munja))
print(ord('A'))
print(ord('a'))
print(ord('0'))


