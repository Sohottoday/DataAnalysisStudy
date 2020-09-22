

# ZeroDivisionError

try:
    x = 4
    y = 0

    mydict = {'a':10}
    print(mydict['b'])

    mylist = [1, 2, 3]
    print(mylist[4])
    z = x / y
    print(z)
except ZeroDivisionError as err:
    print('0으로 나누시면 안됩니다.')
    print(err)
except IndexError as err:
    print('인덱스 범위 관련 오류 발생')        # 위의 코드에서 인덱스 오류가 먼저 발생하므로 해당 문자열이 출력된다.
    print(err)
except KeyError as err:
    print('사전에 해당 키가 없습니다.')
    print('찾고자 하는 키')
    print(err)
except Exception as err:
    print('기타 나머지 예외 발생')
    print(err)
else:
    print('예외가 없으면 이 라인이 실행됩니다.')
finally:
    print('예외 발생 여부아 상관 없이 무조건 실행됩니다.')

'''
예외 처리 : 예외가 사전에 발생하지 않도록 막음 조치를 취하는 것
ZeroDivisionError : 0으로 나누고자 했을때
IndexError : 인덱스 범위를 초과하여 접근시도 시 발생
KeyError : 사전에 해당 키가 존재하지 않을 때

예외 처리 방법
try:
    일반적인 코드 작성
except 예외클래스이름 [as 예외별칭]:
    적당한 오류 메세지 보여주기
else:
    예외가 없을 때 하고자 하는 내용 기록
finally:
    예외 발생 여부와 상관 없이 하고자 하는 내용 기록
    주로 마감 작업(파일 닫기, 데이터베이스 접속을 종료
'''