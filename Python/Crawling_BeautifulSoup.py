# 크롤링 : 웹페이지에서 필요한 데이터를 추출해 내는 작업
# BeautifulSoup : 크롤링을 위한 라이브러리

# 크롤링을 위해서는 HTML5와 CSS3에 대한 기본적인 이해도가 있어야 한다.

# 파이썬 폴더의 scripts 로 이동하여 cmd 창을 띄운다.
# pip list -> 설치된 라이브러리 목록
# pip3 install BeautifulSoup4     -> 크롤링 라이브러리 설치
# pip3 install requests

import urllib.request
from bs4 import BeautifulSoup

page = urllib.request.urlopen('http://ksp.credu.com/ksp/servlet/controller.gate.course.CourseListServlet?p_process=select-course&p_grcode=000002&p_menucd=M0011&p_field=101&p_leftid=101000000000000&p_searchgubun=A')

#검색이 용이한 soup객체를 생성합니다. 
#Break Point 
soup = BeautifulSoup(page, 'html.parser')

#<td>태그를 검색한다. 
print(soup.find_all("td"))