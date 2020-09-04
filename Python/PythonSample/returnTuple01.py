
# 튜플을 사용하여 반환되는 데이터를 여러개 만들기
def myfunc(su1, su2):
    if su2 == 0:
        temp = su1
    else:
        temp = su1//su2

    return su1+su2, su1-su2, su1*su2, temp


su1 = 14
su2 = 5
result = myfunc(su1, su2)
print(result)
print('-'*20)

# 리스트의 모든 요소의 절대값을 구하고, 초대, 최소, 총합, 평균을 튜플로 반환
def myfunc2(mylist):
    mylist = [abs(item) for item in mylist]
    return max(mylist), min(mylist), sum(mylist), sum(mylist)/len(mylist)


mylist = [10, -120, 30, -50, 40]
result = myfunc2(mylist)
print(result)
print('-'*20)


