# private : 클래스 내부의 멤버 변수 중 숨기고 싶은 변수
# 특징 : 클래스 내부 변수는 일반적으로 public속성을 갖기 때문에 외부에서 마음대로 접근하거나 변경할 수 있음
# 이름 변경(Naming Mangling)
#   - 외부에서 접근이 어렵도록 하는 파이썬의 특징
#   - 외부에서 클래스 내부의 멤버 변수를 호출할 때, 원래 이름이 아닌 _클래스명__멤버 변수로 변경됨
# _name    =>    _클래스 명__name

# 식별자 : 키워드는 아니지만 private 멤버 변수로 사용하기 위해, 미리 정해진 용도로 사용하는 문자
# _* : 모듈(파일) 안에서 _로 시작하는 식별자를 정의하면 다른 파일에서 접근할 수 없음
#       ex) _name
# __*__ : 식별자의 앞뒤에 __가 붙어 있는 식별자는 시스템에서 정의한 이름
#       ex) __doc__
# __* : 클래스 안에서 외부로 노출되지 않는 식별자로 인식
#       ex) __name
class BankAccount:
    __id = 0
    __name = ""
    __balance = 0       

    def __init__(self, id, name, balance):
        self.__id = id
        self.__name = name
        self.__balance = balance

    def deposit(self, amount):
        self.__balance += amount

    def withdraw(self, amount):
        self.__balance -= amount
    
    def __str__(self):
        return "{0},{1},{2}".format(self.__id, self.__name, self.__balance)

account1 = BankAccount(100, "전우치", 15000)
account1.withdraw(3000)
print(account1)

# print(account1.__balance)    => 에러 발생! = 원래 내부 이름으로 접근하면 접근하지 못한다.

print(account1._BankAccount__balance)
account1._BankAccount__balance = 35000
BankAccount._BankAccount__balance = 35000

