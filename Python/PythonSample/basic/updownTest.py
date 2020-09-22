
import random
# for idx in range(1, 11):
#     answer = random.randint(1, 100)
#     print(answer)

answer = random.randint(1, 100)
print('정답 : %d' % (answer))
cnt = 0

while True:
    # su : 우리가 입력한 숫자
    su = int(input('1부터 100사이의 정수 1개 입력 : '))
    cnt += 1        # 시도 횟수 증가
    if answer > su:
        print('%d보다 큰 수를 입력하세요.' % (su))
    elif answer < su:
        print('%d보다 작은 수를 입력하세요.' % (su))
    else:
        print('정답입니다.')
        print('%d번만에 맞췄습니다.' % (cnt))
        break

print('finished')
