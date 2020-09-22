
# tuple을 저장하고 있는 list 자료 구조
examdata = [(50, 70), (60, 75), (55, 80)]

for (midexam, finalexam) in examdata:
    print('중간 고사 : %d, 기말 고사 : %d' % (midexam, finalexam))

print('-' * 30)

print('고사별 총합 구해보기')
midsum = finalsum = 0
for (midexam, finalexam) in examdata:
    midsum += midexam
    finalsum += finalexam
print('중간 총합 : %d, 기말 총합 : %d' % (midsum, finalsum))

print('-' * 30)

print(examdata[1][1])

print('range() 함수를 사용하여 고사별 총합 구해보기')
midsum = finalsum = 0
for idx in range(0, 3):
    midsum += examdata[idx][0]
    finalsum += examdata[idx][1]

print('중간 총합 : %d, 기말 총합 : %d' % (midsum, finalsum))
print('-' * 30)





