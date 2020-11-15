from bs4 import BeautifulSoup
import urllib.request as req

# 웹 문서 가져오기
url = "https://finance.naver.com/marketindex/"

res = req.urlopen(url)

soup = BeautifulSoup(res, 'html.parser')
usd_dollar = soup.select_one("div.head_info > span.value").string
print("usd/krw = ", usd_dollar)




url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%ED%95%9C%EC%9A%A9%EC%9A%B4"
res = req.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')

si_list = soup.select("#mw-content-text > div > ul > li > a")

for a in si_list:
    name = a.string
    print("*", name)

# soup.select_one("#vegetable > li:nth-of-type(2)").string          li 중 2번째를 추출
# soup.select_one("li:nth-of-type(6)").string          li 중 6번째를 추출
# soup.select("#fruits > li.yellow")[1].string          여러개의 id나 class가 적용되있다면 이런식으로 list형식으로 추출

"""
find 메서드를 이용해서 추출
condition = {"data-lo":"us", "class":"red"}
print(soup.find("li", condition).string)
"""