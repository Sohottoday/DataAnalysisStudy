
# 계산기
# 계산기 A : 3+4=7
# 계산기 B : 5+15=20

# step1 : 클래스(템플릿) 정의
# step2 : 객체 생성
# step3 : 변수에 값을 할당
# step4 : 출력이나 다른 용도로 사용

# 클래스 구성 요소 : 변수, 함수, 생성자(__init__)
# self 키워드 : 호출하고 있는 객체의 정보가 복사되는 변수
# 클래스 내부에서 함수의 첫 번째 매개 변수로 사용되어야 한다.
class Calculator:
    def __init__(self, data):       # 생성자 : 내부에 들어있는 변수들의 초기화 용도
        self.result = 0     # 총합을 0값으로 초기화
        self.data = data
        print(self.data + '계산기가 생성되었습니다.')
        print('계산기 초기 값 : ', self.result)

    def calc(self, num):
        self.result += num
        return self.result

# 객체 생성 : 해당 클래스를 이용하여 물건 생성
# Calculator는 생성자가 되고, 생성자의 이름은 클래스의 이름과 동일해야 한다.
cal1 = Calculator('계산기 A')
print(cal1.calc(3))
print(cal1.calc(4))
print(cal1.calc(3))

cal2 = Calculator('계산기 B')
print(cal2.calc(5))
print(cal2.calc(15))

print('finished')