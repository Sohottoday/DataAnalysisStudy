"""
# XML 데이터
- XML(Extensible Markup Language) : 특정 목적에 따라 데이터를 태그로 감싸 마크업하는 일반적인 형식의 언어. W3C에 의해 만들어졌다.
- 텍스트 데이터를 기반으로 한 형식. 웹 API에서 많이 사용되는 형식 중의 하나
- XML은 데이터를 계층구조(트리구조, hTML의 DOM구조)로 표현할 수 있다는 특징이 있다.

기본 형식
<요소 속성 = "속성값"> 내용 </요소>
원하는 요소이름을 아무거나 사용할 수 있다.
하나의 요소에는 여러개의 속성을 추가해도 상관없다.

<products type="a">
    <product id="a001" price="1000"> a상품 </product>
</products>

위와 같이 XML은 계층구조로 만들어서 복잡한 데이터를 표현할 수 있다.

"""


# https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp

import urllib.request as req
import urllib.parse as parse         # url을 인코딩하기 위해 불러오는 모듈

rssUrl = "https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"

# 매개변수 지역별 코드를 지정하는 변수
values = {
    'stnId' : '108'         # 108은 전국이라는 의미, 109는 서울/경기도, 이런식으로 지역 코드가 존재한다.
}

params = parse.urlencode(values)

url = rssUrl + "?" + params

print("url = ", url)

data = req.urlopen(url).read()
text = data.decode("utf-8")

# print(text)

import sys

# 명령줄 인수가 제대로 입력되었는지 확인
if len(sys.argv) <= 1:      #명령줄 인수가 1 이하이면 오류 메세지 출력
    print("사용법 : python 인수1 인수2")
    sys.exit()

regionCode = sys.argv[1]

values = {
    'stnId' : regionCode
}

params = parse.urlencode(values)

url = rssUrl + "?" + params

print("url = ", url)

# RSS 데이터를 다운로드
data = req.urlopen(url).read()
text = data.decode("utf-8")



from bs4 import BeautifulSoup

import os.path

url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"

fileName = "forecast.xml"
print(fileName)

if not os.path.exists(fileName):
    req.urlretrieve(url, fileName)
    

# 다운받은 파일을 분석하기
xml_data = open(fileName, "r", encoding="utf-8").read()

soup = BeautifulSoup(xml_data, 'html.parser')

# 날씨에 따라 지역 분류해보기
info = {}
for location in soup.find_all("location"):
    cityName = location.find("city").string
    weather = location.find("wf").string
    if not(weather in info):
        info[weather] = []

    info[weather].append(cityName)

# 지역의 날씨를 구분해서 분류하기
for weather in info.keys():
    print("**", weather)
    for name in info[weather]:
        print(" - ", name)


