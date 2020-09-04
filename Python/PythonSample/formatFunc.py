
su = 3
fruit = '사과'

print('positional argument')    # 인덱스 기반 매개 변수 대입
str1 = '나는 {}를 {}개 먹었습니다.'
print(str1.format(fruit, su))

str2 = '나는 {0}를 {1}개 먹었습니다.'
print(str2.format(fruit, su))

str3 = '나는 {1}를 {0}개 먹었습니다.'
print(str2.format(fruit, su))

print('keyword argument')
str4 = '나는 {abc}를 {defg}개 먹었습니다.'
print(str4.format(defg=su, abc = fruit))

# 2가지 혼합 방식
str5 = '나는 {abc}를 {}개 먹었습니다.'
print(str5.format(su, abc=fruit))

# positional argument가 먼저 와야한다.
# str6 = '나는 {abc}를 {}개 먹었습니다.'
# print(str6.format(abc=fruit, su)) <--

name = '김철수'
fruit = '사과'
su1 = 8

# 서식 지정자 : %s(string), %d(decimal), %f(float_실수), %c(문자 1개), %o(8진수), %x(16진수), %%(%문자를 표현하고자 할 때 사용)
myformat = '%s가 %s를 %d개 먹었습니다.'
print(myformat % (name, fruit, su1))

su1 = 4
su2 = 9
# 4곱하기 9는 36입니다.
suformat = '%d 곱하기 %d는 %d입니다.'
print(suformat % (su1, su2, (su1*su2)))

# pow(a, b) : a의 b제곱
print(pow(5 ,2))

# 2.0의 10.0승은 1024.0 입니다.
su1 = 2.0
su2 = 10.0
suformat = '%.1f의 %.1f 승은 %.1f입니다.'
print(suformat % (su1, su2, pow(su1, su2)))

rate = 0.4567
print('비율 : %.3f%%' % (100*rate))