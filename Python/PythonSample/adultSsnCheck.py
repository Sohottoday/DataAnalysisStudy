
name = '김철수'
ssn = '121225-3456789'

# 슬라이싱과 if구문 등을 이용하여 다음과 같이 출력해보세요.
# 19세 이상이면 성년
'''
출력 결과
이름 : 김철수
주민 번호 : 121225-3456789
성별 : 남자
나이 : 8세
체크 : 미성년자
'''

se = ''
if ssn[7] == '1' or ssn[7] == '3':
    se = '남자'
else:
    se = '여자'

age = 20 - int(ssn[:2])

check = ''
if age >= 19:
    check = '성인'
else:
    check = '미성년자'

print(f"""
출력 결과
이름 : {name}
주민 번호 : {ssn}
성별 : {se}
나이 : {age}세
체크 : {check}
""")

# 강사님 답안
dpos1 = ssn[7]
if dpos1 in ['1', '3']:
    gender = '남자'
else:
    gender = '여자'