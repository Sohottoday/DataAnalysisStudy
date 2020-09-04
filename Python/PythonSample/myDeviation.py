import math

mylist = [10, 30, 40, 80]

# 평균을 구합니다.
# 평균 = 총합(160)/요소갯수 = 160/4 = 40

# (점수-평균)을 제곱을 모두 누적시킵니다.
# 900+100+0+1600+2600

# 위 결과에 도수를 나눕니다.
# 2600/요소갯수 = 2600/4 = 650

# 위 결과에 루트를 씌웁니다.
# 루트 650 = 25.4950975
# sqrt(650)
# 루트는 외부 모듈인 math 모듈의 sqrt를 사용한다.

def pyojun(n):
    avg = sum(n)/len(n)
    for i in range(len(n)):
        n[i] -= avg
        n[i] = n[i]**2
    res = sum(n) / len(n)
    result = math.sqrt(res)
    print('해당 데이터의 표준편차는 %.6f 입니다.' % (result))


pyojun(mylist)
