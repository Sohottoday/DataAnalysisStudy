# 함수의 마지막에는 return 구문이 들어간다.
# 만약 명시하지 않으면 return None라는 구문이 숨어있따.

def namePrint(name, age):
    print(f'{name}님의 나이는 {age}살입니다.')
    return None

name = '제시카'
age = 20
result = namePrint(name, age)
print(result)

if result == None:
    print('데이터를 구하지 못했습니다.')
else:
    print('참 잘했어요')

print('-' * 20)

def gugudan(su):
    rng = range(1, 10)
    for i in rng:
        print('{} * {} = {}'.format(su, i, (su*i)))

dan = 3
gugudan(dan)
print('-' * 20)

# 2, 4, 7단을 출력하시오
dan0 = [2, 4, 7]
for num in dan0:
    gugudan(num)

# 2단이면 [2, 4, 6 ... 18] 출력되는 함수를 만들어보세요.
def Gugu(n):
    gulist = []
    for i in range(1, 10):
        gulist.append(n * i)
    # result = [n * idx for idx in rangae(1, 10)]
    print(gulist)

su = 2
Gugu(su)

# 2단부터 4단까지 각 단의 결과를 list형식으로 출력

newlist = [2, 3, 4]
for n in range(0, len(newlist)):
    Gugu(newlist[n])
