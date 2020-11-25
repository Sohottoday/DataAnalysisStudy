# JSON 데이터
# - 텍스트 데이터를 기반으로 한 데이터 형식
# - JSON(JavaScript Object Notation) : 자바스크립트에서 사용하는 객체 표기방법
# - 자바스크립트 전용 데이터 형식이 아니다. 다양한 소프트웨어와 프로그래밍 언어에서 많이 사용되는 형식으로 데이터 교환할 때 xml보다 가볍기 때문에 최근 선호하는 형식이다.
# - 확장자는 ".json"

# - 구조가 단순하다는 것이 큰 장점.
# - 수많은 프로그램이 언어에서 인코딩/디코딩 표준으로 JSON을 지원하고 있다.
# - 파이썬 모듈에도 JSON 모듈을 지원하고 있다.
# - 최근 웹 API들이 JSON형식으로 데이터를 제공하고 있다.

# JSON의 구조
# - 자료형은 숫자, 문자열, 불린값, 배열, 객체 null이라는 6가지 종류를 사용할 수 있다.
# ex) 40, "string", true/false, [1, 2, 4], {"key":"value", "key":"value"...}, null

import json

price = {
    "time":"17-01-02",
    "price":{
        "apple":1000,
        "banana":3000,
        "orange":2000
    }
}

# json.dumps 메서드는 json 형식으로 출력한다.
jsonData = json.dumps(price)
print(jsonData)

import urllib.request as req
import os.path

# json 데이터 다운로드하기
url = "https://api.github.com/repositories"
fileName = "rep.json"

if not os.path.exists(url):
    req.urlretrieve(url, fileName)

jsonData1 = open(fileName, "r", encoding="utf-8").read()
data = json.loads(jsonData1)

for dat in data:
    print(dat["name"] + " - " + dat["owner"]["login"])
