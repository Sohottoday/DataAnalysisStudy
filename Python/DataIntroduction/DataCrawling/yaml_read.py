# # YAML 데이터 분석
# - YAML은 들여쓰기를 사용해 계층 구조를 표현하는 것이 특징인 데이터 형식
# - XML보다 간단하며, JSON과 거의 비슷하다.
# - YAML은 JSON 대용으로 많이 사용되기도 하고, 애플리케이션 설정 파일을 작성할 때 많이 사용된다
# - 웹 프레임워크인 Ruby와 Symfony(PHP)의 설정 파일 형식으로 사용되고 있다.
# - 파이썬/PHP/루비 등을 포함한 여러 프로그래밍 언어에서 YAML형식을 다루기 위한 라이브러리가 제공되고 있다.
# - 파이썬에서 YAML을 다루기 위해서 PyYAML이라는 모듈을 설치해야 한다.
# >pip3 install pyyaml

# # YAML 데이터 형식
# YAML의 기본은 배열, 해시, 스칼라(문자열, 숫자, 불리언)
# 배열을 나타낼 때는 하이픈(-)을 붙여서 사용한다. 하이픈 뒤에는 공백을 사용해야 한다.
# - apple
# - orange
# - banana
# 중첩 배열(배열안의 배열)
# - yellow
#     - orrange
#     -banana
# - orange
# - red
#     - apple
#     - strawberry

# 해시 표현방법 : 해시는 자바스크립트의 객체와 같은 것을 의미("키":"값")
# name:hong
# age:24
# color:white
# 해시의 계층구조 표현 방법
# name:kim
# property:
#     age:10
#     color:brown
# 배열과 해시를 조합하면 조금 더 복잡한 데이터를 표현할 수 있다.
# - name:lee
#   color:brown
#   age:20
#   hobby:
#     - sports
#     - movie
# - name:smith
#   color:white
#   age:15
#   hobby:
#     - music
#     - tennis

# YAML은 플로우 스타일(Flow Style)이 지원된다.
# 플로우 스타일은 한줄로 표현할 수 있는 방법을 말한다.
# {key:value1, key2:value2, ...}
# 이때 쉼표(,)와 콜론(:) 뒤에는 반드시 공백이 있어야 한다.
# - name:lee
#     hobby: ["sport", "watching movies"]
# - name:smith
#     hobby: ["music", "reading"]

# YAML은 주석을 사용할 수 있다. 주석 기호는 "#"으로 시작한다.

# 여러줄의 문자열을 지정할 수 있다.
# multi-line: |               # \ + shift
#     I am a boy.
#     I am a student.
#     I like Orange.

# YAML의 앵커와 별칭(alias)
#     &aaa1    : 일종의 변수 선언 <-- 앵커라고 한다.
#     *aaa1   : 별칭

# &aaa1 "bbbb"
# *aaa1은 &aaa1의 별칭이다.

import yaml

yaml_data = """
color_def:
    - &col1 "#ff0000"
    - &col2 "#00ff00"
    - &col3 "#0000ff"

color:
    title: *col1
    title2: *col2
    title3: *col3
"""

data = yaml.load(yaml_data)

# 별칭을 이용한 출력
print("title1=", data["color"]["title1"])
print("title2=", data["color"]["title2"])
print("title3=", data["color"]["title3"])




######
# yaml 데이터 정의
yaml_data2 = """
data:2017-01-01
productList:
    -
        id: 100
        name: banana
        color: yellow
        price: 1000
    -
        id: 200
        name: orange
        color: orange
        price: 700
    - 
        id: 300
        name: apple
        color: red
        price: 1200
"""