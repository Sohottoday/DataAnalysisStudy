
# minval : 튜플 요소에서 가장 큰 수와 가장 작은 수를 반환하는 함수 만들기
def minval(*args):
    resmax = 0
    resmin = 0
    for item in range(len(args)):
        if args[item] > args[item-1]:
            resmax = args[item]

        if args[item] < args[item-1]:
            resmin = args[item]

    return f'최소값 : {resmin}, 최대값 : {resmax}'


print(minval(3, 5, 8, 2))

'''
강사님 코드
def minval(*args):
    mymin = min(args)
    mymax = max(args)
    return mymin, mymax
혹은
def minval(*args):
    mylist = [item for item in args]
    mylist.sort()
    return mylist[0], mylist[-1]
'''