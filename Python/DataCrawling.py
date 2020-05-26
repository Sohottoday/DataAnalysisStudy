import urllib.request
from bs4 import BeautifulSoup

data = urllib.request.urlopen('http://ksp.credu.com/ksp/servlet/controller.gate.course.CourseListServlet?p_process=select-course&p_grcode=000002&p_menucd=M0011&p_field=101&p_leftid=101000000000000&p_searchgubun=A')
#검색이 용이한 soup객체를 생성합니다. 
soup = BeautifulSoup(data, 'html.parser')

#<td>태그 중에 class="title"로 된 태그를 검색합니다. 
#<td class="title">
#  <a href="/webtoon/detail..." onclick="click...">1144. 학교</a>
#</td>
#creduList= soup.find_all("td", class_="title")
creduList= soup.find_all('td', attrs={'class':'title'})

#title = cartoons[0].find('a').text
#link = cartoons[0].find('a')['href']
#print(title)
#print(link)
#리스트를 출력하기 위해서 반복문을 사용합니다.
# for item in creduList:
#     aTag = item.find('a')
#     print(aTag.get_text().strip().replace("\n", ""))

for item in creduList:
    try:
        title = item.find('a').get_text()
        print(title.strip())
    except:
        pass
