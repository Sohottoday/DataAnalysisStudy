# 참과 거짓

# 숫자 : 0은 거짓, 0이 아니면 참
# 문자열 : 비어있으면 거짓, 1글자 이상이면 참
# 리스트나 튜블은 0개이면 거짓

su = 1
if su:
    print('참')
else:
    print('거짓')

mylist = [1, 2, 3]
if mylist:
    print('리스트 참')
else:
    print('리스트 거짓')

mytuple = ()
if mytuple:
    print('튜플 참')
else:
    print('튜플 거짓')

