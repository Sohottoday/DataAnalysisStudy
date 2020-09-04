
str1 = 'Hello Korea'

print(str1[0])
print(str1[1], str1[6])
# 마이너스가 나오는 경우 뒷쪽에서 카운트 하되 1base가 된다.
print(str1[-3])

# KHa를 출력해보세요.
print(str1[6], str1[0], str1[-1], sep='')

# 슬라이싱 : 전체 내용 중에서 일부 내용을 연속적으로 추출하는 것
# [from : to : step] : step의 기본값은 1
# from <= 슬라이싱 < to
# step은 몇번째마다 추출하라는 의미 -> ex step 2 는 0, 2, 4, 6 이런식으로 추출하라는 의미
# step의 default값은 1이다.
print(str1[0:5])
print(str1[:5])
print(str1[0:5:2])

ssn = '881120-1234567'

apos = ssn[0:6]
print('앞자리 : ', apos)

# 뒷자리 추출
back = ssn[7:14]
back1 = ssn[7:]
print('뒷자리 : ', back)

aa = ssn[7]
if aa == '1' or aa == '3':
    print('남')
else:
    print('여')

rainbow = ['빨', '주', '노', '초', '파', '남', '보']
another = rainbow[4:7]
print(another)

even = rainbow[0:7:2]
print(even)    # 짝수번째만 뽑아올 때

odd = rainbow[1:7:2]
print(odd)    # 홀수번째만 뽑아올 때

abcd = rainbow[-3 : -1]
print(abcd)

