
# 무한 whileLoop : 반복 횟수가 몇번인지 모르는 경우
# 어느 조건을 충족하면 break 구믄을 사용하여 반드시 종료해야 한다.
# cnt = 0
# while True:
#     print('a' + str(cnt))
#     cnt += 1
#     if cnt == 5:
#         break

# 사용자가 입력한 시험 점수에 대한 평균과 개수를 구해봅니다.
# 음수 값이 입력되면 프로그램을 종료하도록 합니다.

total = 0   # 총점
cnt = 0     # 시험 본 횟수
avg = 0.00  # 평균 점수

while True:
    jumsu = int(input('시험 점수를 입력하세요 : '))
    if jumsu < 0:
        break
    total += jumsu
    cnt += 1
    avg = total/cnt

print(f'총점 : {total}, 횟수 : {cnt}, 평균 : {avg}')

print('finished')