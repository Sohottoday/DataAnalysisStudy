
# 매개 변수에 별 2개는 dict(사전)으로 인식한다.
def myfunction(a, b=0, *args, **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('튜플 출력')
    for item in args:
        if type(item) == str:
            print('문자열 : ', item)
        elif type(item) == int:
            print('정수 : ', item)
        elif type(item) == float:
            print('실수 : ', item)
        else:
            print('기타 : ', item)

    print('-'*30)
    print('사전 출력')
    for key, value in kwargs.items():
        print(f'키 : {key}, 값 : {value}')
    print('-' * 30)

# myfunction(10)
myfunction(10, 20, 30, 'abc', 12.3456, 50, kim=(40, 50), park=30)
