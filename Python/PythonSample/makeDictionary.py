
mydict = {'김철수':35, '박영희':50, '홍길동':40}
print(mydict)

for key in mydict.keys():
    print(key)

print('-' * 30)

for value in mydict.values():
    print(value)

print('-' * 30)

for key in mydict.keys():
    print('{}의 나이는 {}살 입니다.'.format(key, mydict[key]))

print('-' * 30)

for name, age in mydict.items():            # dict 의 key와 value를 한번에 다 가져오는 함수
    print('{}의 나이는 {}살 입니다.'.format(name, age))

print('-' * 30)

findkey = '심형래'
if findkey in mydict:
    print(findkey + '은 존재함')
else:
    print(findkey + '은 존재하지 않음')

result = mydict.pop('홍길동')
print('pop 이후 내용 :', mydict )
print('pop 된 내용 :', result )

mydict.clear()
print(mydict)
print('-' * 30)
