# 상속(Inheritance)과 다형성(Override))

# 상속 : 부모 클래스의 모든 멤버를 자식에게 물려줄 수 있는 것
# 다형성 (Override) : 상속받은 메소드의 바디를 덮어 쓰기

class Person:
    def __init__(self, name, phoneNumber):
        self.name = name
        self.phoneNumber = phoneNumber

    def printInfo(self):
        print("Info(Name:{0}, Phone Number : {1}".format(self.name, self.phoneNumber))

class Student(Person):
    def __init__(self, name, phoneNumber, subject, studentID):
        #self.name = name
        #self.phoneNumber = phoneNumber
        Person.__init__(self, name, phoneNumber)        # 위와 같이 직접 설정해줘도 되고 이런식으로 명시적으로 불러올 수도 있다. 값은 같다.
        self.subject = subject
        self.studentID = studentID

    def printInfo(self):
        print("Info(Name:{0}, Phone Number : {1}".format(self.name, self.phoneNumber))
        print("Info(Subject:{0}, StudentID:{1}".format(self.subject, self.studentID))

p = Person("전우치", "010-222-1234")
s = Student("이순신", "010-111-1234", "컴공", "991122")
#print(p.__dict__)
#print(s.__dict__)
p.printInfo()
s.printInfo()
