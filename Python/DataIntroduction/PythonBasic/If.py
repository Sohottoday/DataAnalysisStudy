# 조건 구문 규칙
# 가장 바깥쪽에 있는 블록의 코드는 반드시 1열부터 시작
# 내부 블록은 같은 거리만큼 들여쓰기
# 블록의 끝은 들여 쓰기가 끝나는 부분으로 간주됨 {,},begin,end 를 사용하지 않음
# 탭과 공백은 섞어서 쓰지 않는 것이 좋음
# 일반적으로 공백은 2 ~ 4칸을 사용

# if문의 정의 : if구문은 조건에 따라 분기를 할 경우 사용
# if 구문 다음의 조건식이 참이면 if 안쪽을 수행
# if 구문 다음의 조건식이 거짓이면 else 구문을 수행

# if <조건식>:
#   <구문>      # if구문만 단독으로 사용하는 경우 참인 경우만 처리. 거짓에 대한 분기는 따로 하지 않음.

# if <조건식 1>:        # 조건이 여러 개 인 경우
#   <구문 1>
# elif <조건식 2>:      # if 첫번째 조건식인 구문1이 false면 elif를 수행
#   <구문 2>
# else:
#   <구문 3>            # 구문 1과 2 모드 false인 경우, else 구문으로 처리

# if else문 : 조건을 평가해서 참인 경우 거짓인 경우를 처리
score = int(input('InputScore:'))       # input()함수를 통해 점수를 입력 받는 경우
if 90 <= score <=100:                   # input() 함수는 무조건 문자열 형식을 리턴
    grade = "A"                         # int()함수로 감싸서 숫자형식으로 변환한 결과를 score변수에 저장
elif 80 <= score < 90:
    grade = "B"
elif 70 <= score < 80:
    grade = "C"
elif 60 <= score < 70:
    grade = "D"
else:
    grade = "F"
print("Grade is " + grade)

# 판단 방법
# 파이썬은 어떤 값을 논리식에서 비교할 때 참 또는 거짓으로 판단하는 근거를 가지고 있음.
# 정수 계열의 0, 실수 계열의 0.0, 시퀀스 계열의 (), [], {}, "", None은 항상 거짓으로 판단
print(bool(True))
print(bool(False))
print(bool(13))
print(bool(0.0))
print(bool('apple'))        # 문자열형식은 빈문자열 ""이면 False, "abc"와 같이 문자가 있으면 True
print(bool(""))
print(bool(()))
print(bool([10,20]))

