
# 정수 2개를 입력받아서, 앞수에서 뒤수 사이에 있는 모든 정수의 합을 구하시오.
# 앞수 : 2, 뒤수 : 4이면 2+3+4=9 출력
# 앞수 : 5, 뒤수 : 2 이면 5+4+3+2=14 출력

# 출력 예시 : 2부터 4까지의 총합은 9입니다.

su1 = int(input('첫번째 숫자를 입력해주세요 : '))
su2 = int(input('두번째 숫자를 입력해주세요 : '))
result = 0

if su1 < su2:
    for num in range(su1, su2+1):
        result += num
    print(f'{su1}부터 {su2}까지의 총합은 {result}입니다.')
elif su1 > su2:
    for num in range(su2, su1+1):
        result += num
    print(f'{su2}부터 {su1}까지의 총합은 {result}입니다.')
else:
    result = su1
    print(f'{su1}부터 {su2}까지의 총합은 {result}입니다.')




