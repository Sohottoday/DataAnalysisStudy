# 디버깅(Debugging) : 오류가 발생되는 경우에 라인 단위로 추적하고 수정하는 작업

# Visual Studio Code 디버깅 기능    => 디버깅 시작
# 중지점(break point) 추가  => 라인 숫자 앞 빨간 점
# 실행버튼(F5) 클릭
# 한단계 씩 코드 실행(F11)

def add(x,y):
    return x+y
#Break Point
result = add(3,4)
print(result)
