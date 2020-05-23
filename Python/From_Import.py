# 윈도우 환경 변수 동적으로 추가하는 방법
# sys.path.append() 로 경로 추가 가능
# sys.path.remove()로 경로 삭제 가능
import sys
print(sys.path)

# from : 특정 모듈을 지정해서 메모리에 탑재할 때 사용하는 구문
# import : 특정 함수를 지정해서 메모리에 탑재할 때 사용하는 구문

# from <모듈> import <어트리뷰트>
# 임포트된 어트리뷰트는 '모듈이름.어트리뷰트이름' 같은 형식으로 쓰지 않고 바로 사용 가능
#from simpleset import union        -> simpleset 클래스에서 union 함수만 탑제하겠다 라는 의미. -> 메모리에 simpleset이 전부 탑재되는 것이 아니라 union함수만 탑재된다.
#union([1,2,3],[3],[3,4])
#=> [1,2,3,4]

# intersect([1,2],[1])      -> 메모리에 탑재 된 적이 없기 때문에 찾지 못한다고 뜬다.
# => Traceback (most recent call last):
#   File "<pyshell#25>", line 1, in <module>
#        intersect([1,2],[1])
#   NameError : name 'intersect' is not defined

# from <모듈> import *
# 모듈 내 이름 중 _로 시작하는 어트리뷰트를 제외하고 모든 어트리뷰트를 현재의 이름 공간으로 임포트 가능

# from simpleset import *

# 모듈은 메모리에 한 번만 로딩
# 참조하는 별칭은 여러 개 있을 수 있음(다른 이름으로 여러번 참조 되어도 한 번만 로딩)

# import testmodule as test1
# import testmodule as test2    <- 이름은 다르지만 동일한 것 참조.
# test1. defaultvalue = 100
# test2.printDefaultValue()    => 100 출력