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

