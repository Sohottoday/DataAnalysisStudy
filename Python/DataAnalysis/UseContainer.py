# 문자열 다루기

word = "HEllo"

# upper() : 문자열을 모두 대문자로
print(word.upper())

# lower() : 문자열을 모두 소문자로
print(word.lower())

# capitalize() : 문자열 첫글자를 대문자로
print(word.capitalize())

# 문자열 정렬
## rjust() : 오른쪽 정렬
## ljust() : 왼쪽 정렬 정렬
## center() : 가운데 정렬
print(word.rjust(9))    # 9칸 기준 오른쪽 정렬
print(word.ljust(9))
print(word.center(9))

# replace() : 문자열 대체, 문자열 변경
print(word.replace('l', 'w'))

print(word.replace('E', '****'))

# strip() : 앞 뒤 공백 모두 제거
word2 = "     hello world         "
print(word2.strip())

# count() : 문자열에서 사용된 문자 또는 단어의 수를 출력
word3 = "good morning good afternoon"
print(word3.count('o'))

print(word3.count('good'))

# find() : 찾고자 하는 문자의 위치를 나타내준다. 인덱스가 아닌 슬라이싱 기준
print(word3.find('morning'))

print(word3.find('good', 3))    # 3번째 위치 이후에서 good의 위치를 찾아라

# join() : 문자열 삽입
print(word3.join("!!"))     # ()안의 문자열 사이에 word3을 넣으라는 의미

# split() : 문자열 나누기

# 표현식 활용
print("{0:>10}".format("hi"))   # 자릿수를 10칸을 준 뒤 오른쪽 정렬
print("{0:^10}".format("hi"))   # 자릿수를 10칸 준 뒤 가운데 정렬
print("{0:=^10}".format("hi"))   # 자릿수를 10칸 준 뒤 문자열은 가운데 정렬 후 공백은 '=' 로 채우라는 의미
print("{0:!<10}".format("hi"))   # 자릿수를 10칸 준 뒤 문자열은 왼쪽 정렬 후 공백은 '!' 로 채우라는 의미


# 배열 다루기
aa = [1.11, 'hello', 33, True]

# append() : 값 추가
aa.append('sorry')
print(aa)

# pop() : 값 제거, 가장 뒤의 값을 제거한다.
aa.pop()
print(aa)

# range() : 정수들로 구성된 리스트를 만드는 함수, 실수로 만들고 싶을때에는 numpy의 arange 활용
number = range(10)
print(list(number))     # 그대로 출력하는게 아닌 list에 담아서 출력해줘야 한다.

# enumerate() : 리스트 각 요소의 인덱스에 접근할 때 사용하는 함수
animals = ['cat', 'lion', 'tiger']
for idx, animal in enumerate(animals):
    print("{0}번째 배열의 값은 {1}입니다".format(idx, animal))

# list comprehension
square = [i**2 for i in number]
print(square)

odd_square = [x ** 2 for x in number if x % 2 == 1]     # 홀수인 값만 제곱하라는 의미
print(odd_square)
