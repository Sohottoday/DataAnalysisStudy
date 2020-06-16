# 정적 메소드 또는 스태틱 메소드로 혼용
# 클래스에서 직접 호출할 수 있는 메소드
# 메소드를 정의할 때 인스턴스 객체를 참조하는 self라는 인자를 선언하지 않음.
# ex) MyCalc.my_add    클래스명.메소드명

# 정적 메소드의 특징
# 클래스 인스턴스에는 적용되지 않는 메소드
# 클래스.메소드 명으로 호출 가능

class MyCalcc(object):
    @staticmethod           # 파이썬에서 사용되는 '데코레이터'라는 문법으로, 메타 데이터를 전달하는 용도로 사용
    def my_add(x,y):
        return x+y

a = MyCalcc.my_add(5,7)
print(a)