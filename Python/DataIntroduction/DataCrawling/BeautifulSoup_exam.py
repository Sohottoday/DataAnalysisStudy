from bs4 import BeautifulSoup

html = """
<html><body>
    <h1 id="title">스크래핑 연습</h1>
    <p id="subtitle">웹페이지를 분석해보기</p>
    <p>데이터 정제하기</p>
    <ul>
        <li><a href="http://www.naver.com">네이버</a></li>
        <li><a href="http://www.daum.net">다음</a></li>
    </ul>
</body>
</html>
"""

# html 분석하기
soup = BeautifulSoup(html, 'html.parser')

# 원하는 요소 접근하기
h1 = soup.html.body.h1
p1 = soup.html.body.p
p2 = p1.next_sibling.next_sibling           # 같은 부모 안에 하나의 태그가 여러가지가 있다면 next_sibling 를 활용해 원하는 요소에 접근이 가능하다.

# 원하는 요소의 내용 추출하기
print("h1 : ", h1.string)
print("p : ", p1.string)
print("p : ", p2.string)

# find() 메서드를 이용한 데이터 추출
title = soup.find(id="title")
subtitle = soup.find(id="subtitle")

print("title = " + title.string)
print("subtitle = " + subtitle.string)

# find_all() 메서드
links = soup.find_all("a")

for a in links:
    href = a.attrs['href']
    text = a.string
    print(text, ">", href)

