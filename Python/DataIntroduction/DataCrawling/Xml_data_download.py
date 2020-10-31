# https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp

import urllib.request
import urllib.parse         # url을 인코딩하기 위해 불러오는 모듈

rssUrl = "https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"

# 매개변수 지역별 코드를 지정하는 변수
values = {
    'stnId' : '108'         # 108은 전국이라는 의미, 109는 서울/경기도, 이런식으로 지역 코드가 존재한다.
}

params = urllib.parse.urlencode(values)

url = rssUrl + "?" + params

print("url = ", url)

data = urllib.request.urlopen(url).read()
text = data.decode("utf-8")

print(text)
