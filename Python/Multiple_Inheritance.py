# 다중상속 : 2개 이상의 클래스를 동시에 상속받는 것을 의미.

class Tiger:
    def jump(self):
        print("호랑이처럼 멀리 점프하기")
    # def cry(self):
    #   print("호랑이 어흥")

class Lion:
    def bite(self):
        print("사저처럼 한입에 꿀꺽하기")
    # def cry(self):
    #   print("사자 어흥")

class Liger(Tiger, Lion):
    def play(self):
        print("라이거만의 사육사와 재미있게 놀기")

l = Liger()
l.jump()
l.bite()
l.play()

# 다중상속의 이름충돌(name conflict)
# 위의 Tiger와 Lion에는 둘 다 cry라는 함수가 정의되어 있는데
# l = Liger()
# l.cry()
# 이런식으로 자식 클래스에서 같은 이름의 함수를 호출하게 되면 위에서부터 호출하여(호출 순서에 의하여)
# "호랑이 어흥" 이라는 Tiger 클래스의 함수가 호출된다.