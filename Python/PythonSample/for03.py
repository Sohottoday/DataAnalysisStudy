
examdata = [90, 30, 65, 45, 80]
print(examdata)
print('-' * 30)

for item in examdata:
    print(item)
print('-' * 30)

# 점수가 60 이상이면 합격, 그렇지 않으면 불합격으로 처리

for idx in range(len(examdata)):
    if examdata[idx] >= 60:
        print(f'{idx+1}번째 응시자 {examdata[idx]}점 : 합격')
    else:
        print(f'{idx+1}번째 응시자 {examdata[idx]}점 : 불합격')

print('-' * 30)

print('합격자만 출력하기')
for idx in range(len(examdata)):
    if examdata[idx] >= 60:
        print(f'{idx+1}번째 응시자 {examdata[idx]}점 : 합격')
    else:
        continue


