"""
# 웹 API(Application Programming Interface) : 어떤 사이트가 가지고 있는 기능을 외부에서 쉽게 사용할 수 있게 공개한 것
# 원래 API는 어떤 프로그램 기능을 외부의 프로글매에서 호출해서 사용할 수 있게 만든 것을 의미한다.
# 간단히 말하자면 서로 다른 프로그램이 기능을 공유할 수 있게 절차나 규약을 정의한 것

# 웹 API를 제공하는 이유
- 두부분의 웹 서비스는 정보를 웹사이트를 통해 제공한다. 이러한 정보는 크롤링의 대상이 된다.
- 수많은 개발자들이 크롤링한다고 하면 서버에 부하가 많이 발생한다.
- 따라서, 크롤링이 될 것이라면 차라리 미리 웹 api를 제공해서 서버의 부담을 줄이는 것이 낫다.
- 결론적으로 웹 api를 제공하는 첫 번째 이유는 서버의 부하를 감소시키는 것
- 두 번째 이유는 상품을 알리거나 판매할 기회를 더 많이 늘리기 위해서(상품 검색 API 제공)

# 웹 API는 유료와 무료로 나뉜다

# 웹 API 사용시 유의사항
- 웹 API 제공자의 문제로 인해 API가 없어지는 경우 사양의 변화가 발생할 수 있다.
- 웹 API를 신뢰할 수 있는지 확인한다. 즉, 지원을 오래할 수 있는지, 수요가 많은지, 큰 기업에서 제공하는지 등
"""

import requests
import json             # 보통 웹 API의 결과는 JSON이나 XML 형식. openweathermap에서는 JSON 형식으로 리턴한다.
# 따라서, JSON 형식의 데이터를 파이썬 데이터형식으로 변환해줘야 하는데 이때 json 모듈을 사용한다.

apikey = "각자의 api키"
city_list = ["Seoul,KR", "Tokyo,JP", "New York,US"]

# API 지정
api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

# 켈빈 온도를 섭씨 온도로 변환하는 함수
k2C = lambda k : k - 273.15

# 각 도시의 정보를 추출하기
for name in city_list:
    pass
# API의 URL 구성하기
url = api.format(city=name, key=apikey)

# API 요청을 보내 날씨 정보를 가져오기
res = requests.get(url)

# JSON형식의 데이터를 파이썬형으로 변환
data = json.loads(res)

# 결과 출력
print("** 도시 = ", data["name"])
print("| 날씨 = ", data["weather"][0]["description"])
print("|최저 기온 = ", k2C(data["main"]["temp_min"]))
print("|최고 기온 = ", k2C(data["main"]["temp_max"]))
