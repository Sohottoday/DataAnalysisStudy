# 가변인자와 정의되지 않은 인자 처리

# 스코핑 룰(Scoping rule)
# 파이썬에서 함수는 별도의 이름 공간(namespace)을 가짐
# 이 때 이름을 찾는 방법을 스코핑 룰이라고 함
# 변수를 사용하기 위해서 반드시 알아야 할 규칙
# 지역(Local) : 함수 내부공간 -> 전역(Global) : 함수 외부 공간 -> 내장(Built-in) : 파이썬 자체에 정의된 공간
# 첫 글자를 하나씩 따서 LGB라고 하며 LGB 순서로 이름을 검색
x = 1
def func(a):
    return a + x        # 함수 내부에 해당 이름이 없기 때문에 전역에서 찾아서 사용
print(func(1))

def func2(a):
    x = 2           # 함수 내부에 x라는 이름이 등록됨
    return a + x
print(func2(1))     # 따라서 x가 2라는 지역변수로 값이 저장되어 있어 내부의 값을 읽어 3을 리턴

g = 1
def testScope(a):
    global g    # 전역변수 g를 함수 내부에서 참조해서 사용
    g = 2       # global g 라는 표기를 통해 전역변수 g를 내부에서 참조한다는 선언을 하면 불변 형식이지만 읽기와 쓰기가 가능.
    return g + a
print(testScope(1))
print(g)

# 인자의 개수가 가변적일 때 어떻게 하는가?
# 1. 기본인자 : 함수를 호출할 때, 인자를 지정해주지 않아도 기본값이 할당되는 방법
def Times(a=10, b=20):
    return a*b

print(Times())      # 전부 기본값을 사용
print(Times(5))     # a 에 5를 전달하고 b는 기본 값을 사용

# 2. 키워드 인자 : 인자 이름으로 값을 전달하는 경우에는 순서를 맞추지 않아도 인자 이름을 지정해서 전달 가능한 방법
def connectURI(server, port):
    str = "http://" + server + ":" + port
    return str

print(connectURI("test.com","8080"))
print(connectURI(port="8080",server="test.com"))

# 3. 가변인자 : *를 함수 인자 앞에 붙이면 정해지지 않은 수의 인자를 받겠다는 의미
def test(*args):
    print(type(args))
test(1,2)

def union(*ar):
    res = []
    for item in ar:
        for x in item:
            if not x in res:
                res.append(x)
    return res
print(union("HAM","EGG","SPAM"))
print(union("gir","generation","gee"))

# 4. 정의되지 않은 인자 처리 : **를 붙이면 정의되지 않은 인자를 사전 형식으로 받을 수 있음
def userURIBuilder(server, port, **user):       # ** 입력이 된다면 딕셔너리 형식으로 입력됨.
    str="http://" + server + ":" + port + "/?"
    for key in user.keys():
        str += key + "=" + user[key] + "&"
    return str
print(userURIBuilder("test.com","8080",id='userid',passwd='1234'))
print(userURIBuilder("test.com","8080",id='userid',passwd='1234',name='mike',age='20'))
