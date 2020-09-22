
month = int(input('달을 입력하세요 : '))

# 입력:9
# 9월은 가을입니다.

if month >= 3 and month <=5:
    print(f'{month}월은 봄입니다.')
elif month >= 6 and month <=8:
    print(f'{month}월은 여름입니다.')
elif month >= 9 and month <=11:
    print(f'{month}월은 가을입니다.')
elif month in [12, 1, 2]:
    print(f'{month}월은 겨울입니다.')
else:
    print('무제')