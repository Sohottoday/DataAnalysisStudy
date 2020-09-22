
# range() 함수 : 반복문에서 특정 횟수만큼 요소들을 반복시키고자 할 때 사용
# range(start, end, stop)

for idx in range(1, 11):
    print(idx)
print('-' * 30)

for idx in range(1, 10, 2):
    print(idx)
print('-' * 30)

for idx in range(10, 1, -1):
    print(idx)
print('-' * 30)

# 1부터 10까지의 총합 구하기
total = 0

for xxx in range(1, 11):
    total += xxx
print('총합 : %d' % (total))

# 1 + 4 + 7 + 10 + ... + 100
dap1 = 0
for num in range(1, 101, 3):
    dap1 += num
print('1번 총합 : ', dap1)

dap2 = 0
for num2 in range(97, 1, -5):
    dap2 += num2
print('2번 총합 : ', dap2)

# 1*1 + 6*6 + 11*11 + ... 96+96 = 63670
result = 0
for idx in range(1, 97, 5):
    result += (idx * idx)
print('3번 총합 : ', result)

'''
강사님 답안
for xxx in range(1, 97, 5):
    total += pow(xxx, 2)
    혹은
    total += xxx ** 2
'''
# abs() : 절대값으로 변경해주는 함수
print(abs(-15))

# 사용자가 숫자를 하나 입력 받고, 1부터 해당 수까지의 총합 구하기
# 숫자 입력 : 5
# 출력 결과 : 1부터 5까지의 합은 15
# 만약 음수값을 입력하면 절대값으로 변경하도록 합니다.
# 숫자 입력 : -5
# 출력 결과 : 1부터 10까지의 합은 55

numin = int(input('숫자 입력 : '))
result = 0
for idx in range(1, abs(numin)+1):
    result += idx
print(result)

'''
강사님 답안
if su <0:
    su = abs(su) 
'''

