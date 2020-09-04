# 주민번호는 반드시 14자리어야 한다.
# 6번째 항목은 반드시 '-' 이어야 한다.
# 2번째 항목은 '0'또는 '1'이어야 한다.
# 7번째 항목은 '0', '1', '2', '3' 이어야 한다.

def findSsn(juminno):
    if len(juminno) != 14:
        return False

    if juminno[6] != '-':
        return False

    #if juminno[2] != '0' or juminno[2] != '1':
    if not juminno[2] in ['0', '1'] :
        return False

    if not juminno[7] in ['0', '1', '2', '3']:
        return False

    if not(juminno[0:6].isdigit()) or not(juminno[7:].isdigit()):
        return False
    return True

# 문자열.isdigit() 함수를 이용하면 숫자들로 구성되었는지 확인이 가능
juminList = ['701226-1710566', '7012261710566', '703226-1710566', '701226-5710566']

for juminno in juminList:
    result = findSsn(juminno)
    if result == True:
        print(f'{juminno}는 올바른 주민 번호')
    else:
        print(f'{juminno}는 잘못된 주민 번호')

print('-'*30)