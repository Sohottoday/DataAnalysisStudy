# 인스턴스 객체 멤버 변수 이름 해석 순서
# 인스턴스객체 내부 -> 클래스 객체 내부 -> 전역 공간

# 두 개의 인스턴스 객체를 생성하여 이름을 해석 하는 방법
class Person:
    name = "Default Name"

p1 = Person()
p2 = Person()
p1.name = "전우치"
print("p1's name", p1.name)
print("p2's name", p2.name)

# 클래스에 새로운 멤버 변수 title 추가하는 방법
Person.title = "New title"
print("p1's title:", p1.title)
print("p2's title:", p2.title)
print("Person's title:", Person.title)

# 인스턴스 객체에 동적으로 멤버 변수를 추가하는 법
p1.age = 20
print("p1's age :", p1.age)
print("p2's age :", p2.age)

# self : 자기 자신을 참조하는 의미
strr = "Not Class Member"
class GString:
    strr = ""
    def set(self, msg):
        self.strr = msg
    def print(self):
        print(self.strr)
g = GString()
g.set("First Message")
g.print()
