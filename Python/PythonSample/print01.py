# print 함수 : 문장을 출력한다.
print('동해물과 백두산이')
print('마르고''닳도록')
print('하느님이', '보우하사', '우리나라만세')

print('안녕하세요', end='@@')        # end 속성의 default 값은 엔터이다.
print('홍길동님')

# input() 함수
# 입력된 데이터는 모두 문자열로 인식한다.
print('이름을 입력하세요')
name = input()
age = input('나이 입력')
print('이름 : ', name, ', 나이 : ', age)

# 데이터 형변환 : 바꿀타입(데이터)
# int는 정수형, str은 문자열
newage = int(age) + 5        # 그냥 사용할 경우 input 타입이 문자열이므로 에러가 뜬다.
# 따라서 문자열을 숫자로 바꿔준 뒤 사용한다.
print('5년뒤 나이 : ' + str(newage))

kor = int(input('국어 점수 입력 : '))
eng = int(input('영어 점수 입력 : '))
math = int(input('수학 점수 입력 : '))

total = kor + eng + math
print('총점 : ', total)
