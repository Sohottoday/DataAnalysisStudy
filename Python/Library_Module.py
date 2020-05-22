# 모듈(Library) : 여러 코드를 묶어 다른 곳에서 재사용 할 수 있는 코드의 모음

# 내장모듈 : 파이썬에서 기본적으로 제공되는 모듈
# 문자열(string), 날짜(date), 시간(time), 수학(math), 랜덤(random), 파일(file)
# sqlite3, os, sys, xml, http등 약 200개 정도
import math

print(math.log(100))
print(math.pi)
print(dir(math))

# 사용자 정의 모듈 : 기본적으로 제공되는 모듈 외 사용자가 직접 작성한 모듈
# 필요시, 모듈 작성 및 제공 가능
# 함수를 구현한 사용자 정의 모듈을 c:\python36\lib 에 복사
from functools import *
def intersect(*ar):
    return reduce(__intersectSC, ar)

def __intersectSC(listX, listY):
    setList = []
    for x in listX:
        if x in listY:
            setList.append(x)
    return setList

def difference(*ar):
    setList = []
    intersectSet = intersect(*ar)
    unionSet = union(*ar)
    for x in unionSet:
        if not x in intersectSet:
            setList.append(x)
    return setList

def union(*ar):
    setList = []
    for item in ar:
        for x in item:
            if not x in setList:
                setList.append(x)
    return setList

# py 이름으로 모듈 제공 하다
# 이 파일의 경우 import Library_Module 로 호출 가능