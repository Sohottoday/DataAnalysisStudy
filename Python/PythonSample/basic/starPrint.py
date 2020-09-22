
def showStar(su=10):
    if su<0:
        su = abs(su)
    print('*'*su)

showStar()
showStar(3)

''' 
강사님 코드
def showStar(su=10)
    for idx in range(su):
        print('*', end='')
    print()
'''

# showStar() 함수를 이용하여 별을 su개 만큼 출력하는 프로그램을 작성하시오.
# 만약 매개변수를 입력하지 않으면 10개를 출력하도록 합니다.

for idx in range(1, 11):
    showStar(idx)
print('-'*20)

showStar(-7)

for item in [3, 5, 9]:
    showStar(item)