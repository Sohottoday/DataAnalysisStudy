# 생성자 메소드 : 객체의 초기화 담당
# 인스턴스 객체가 생성될 때 가장 먼저 호출

# 소멸자 메소드 : 객체의 소멸을 담당
# 인스턴스 객체의 레퍼런스 카운트가 0이 될 때 자동으로 호출

class MyClass:
    def __init__(self, value):      # 생성자 메소드
        self.Value = value
        print("Class is created Value = ",value)
    
    def __del__(self):              # 소멸자 메소드
        print("Class is deleted!")

d = MyClass(10)     # 참조 카운트 1
d_copy = d          # 참조 카운트 2
del d               # 참조 카운트가 1로 감소
del d_copy          # 참조 카운트가 0 -> 이때 소멸자 호출