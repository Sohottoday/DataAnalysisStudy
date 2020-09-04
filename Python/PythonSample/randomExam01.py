# 주사위를 10번 던져서 나온 눈의 총합을 구재후는 jusawee 함수를 만들어 보세요.
# 단, 시도 횟수가 입력되지 않으면 5번을 던진다.
import random

def jusawee(su=5):
    total = 0
    for i in range(0, su):
        randnum = random.randint(1, 6)
        total += randnum
        print(randnum)
    print(total)

sido = 10
jusawee(sido)
jusawee()