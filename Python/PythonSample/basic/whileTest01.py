
# 숫자 5를 입력 받고 5단 출력하기

num = int(input('숫자를 입력하세요 : '))
i = 1

if num < 0:
    abs(num)

while i<10:
    mystring = '%d * %d = %2d' % (num, i, (num*i))
    print(mystring)
    i += 1

print('finished')