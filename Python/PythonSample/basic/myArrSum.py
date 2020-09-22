# 리스트의 모든 요소들의 합을 구해주는 함수 arrsum을 만들어보자



def arrsum(n):
    total = 0
    for item in n:
        total += item
    return total

mylist = [10, 20, 30]
result = arrsum(mylist)
print(result)

mydata = (1, 2, 3)  # 튜플
result = arrsum(mydata)
print(result)

# 집합을 이용하여 테스트
myset = set((11, 22, 33))  # 튜플
result = arrsum(myset)
print(result)