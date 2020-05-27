
from bs4 import BeautifulSoup
import requests
import re 

#블로그에 있는 글의 목록을 가져오기 
req = requests.get('https://steemit.com/@papasmf1')
html = req.text 
soup = BeautifulSoup(html, 'html.parser')

#<h2 class="articles__h2 entry-title" data-reactid="419"><a href="/kr/@papasmf1/5wpai1-4" data-reactid="420"><!-- react-text: 421 -->큐텐에서 공동구매로
blogList = soup.find_all('h2', class_='articles__h2 entry-title')
for item in blogList:
        findItem = item.find('a')
        print(findItem.get_text())
