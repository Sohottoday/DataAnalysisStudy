
# 100번 반복한다.
# 매번 1부터 10사이의 임의의 수를 추출한다.(random 모듈)
# 이것을 사전에 담고, 최종 결과를 출력하도록 한다.('1':2, '2':5 ...)
# list comprehension을 사용하여 리스트로 만든 다음 mylist.sort() 함수를 사용하여 정리

import random
i = 1
mydict = dict()
mylist = list()

for aaa in range(1, 11):
    mydict[aaa] = 0
print(mydict)

while i < 101:
    randnum = random.randint(1, 10)
    i += 1

    if randnum in mydict.keys():
        mydict[randnum] += 1
print(mydict)

for key, value in mydict.items():
    print(f'숫자 {key}는 {value}번 추출')

mylist = [key for key in mydict.keys()]
print(mylist)

# 강사님 코드
'''
mydict = {}

for idx in range(1, 101):
    data = random.randint(1, 11)
    if data in mydict:
        mydict[data] += 1
    else:
        mydict[data] = 1
print(mydict)

mylist = [key for key in mydict.keys()]
mylist.sort()

for key in mylist:
    print('숫자 %d는 %d번 추출' % (key, mydict[key]))
'''