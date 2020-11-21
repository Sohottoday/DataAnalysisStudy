"""
# Selenium : 웹 브라우저를 컨트롤하여 웹 UI를 자동화 하는 도구 중의 하나

# 웹 자동화
 자동화의 종류
 - 가장 원초적인 자동화는 화면의 좌표를 기준으로 한 자동화
 - Selenium 도구를 이용하는 웹 자동화
 - 윈도우즈 자동화
 - 작업의 자동화

Selenium은 Selenium Server와 Selenium Client가 있다.
로컬 컴퓨터의 웹브라우저를 컨트롤하기 위해 Selenium Client를 사용한다.

Selenium Client는 WebDriver라는 공통인터페이스와 각 브라우저 타입별로 웹 드라이버로 구성되어 있다.

웹 드라이버의 구성은
WebDriver.Firefox : 파이어폭스
WebDriver.Chrom : 크롬
WebDriver.Ie : 인터넷 익스플로어
WebDriver.Opera : 오페라
WebDriver.PhantomJS : PhantomJS 브라우저(CLI 형식의 브라우저)

# Selenium Client 설치
>> pip install selenium

# Selenium 사용법
from selenium import webdriver : 사용할 웹드라이버를 import한다.

browser = sebdriver.Chrome('크롬드라이버가 있는 경로')

# 특정 URL을 이용하여 브라우저를 실행시키는 방법
browser.get('https://google.com')

# Selenium으로 DOM요소를 선택하는 방법
- DOM 내부에 있는 여러개의 요소들 중에서 처음 찾아지는 요소를 추출하는 메서드
find_element_by_id(id)
find_element_by_name(name)
find_element_by_css_selector(query)
find_element_by_xpath(query)
find_element_by_tag_name(name)      태그 이름이 name에 해당하는 요소를 하나 추출한다.
find_element_by_link_text(text)             링크 텍스트로 요소를 하나 추출한다.
find_element_by_partial_link_text(text)         링크의 지식요소에 포함돼 있는 텍스트로 요소를 하나 추출한다.

- DOM 내부에 있는 여러개의 요소들을 모두 추출하는 메서드
find_elements_by_css_selector(query)
find_elements_by_xpath(query)
find_elements_by_tag_name(name)
find_elements_by_class_name(name)
find_elements_by_partial_link_text(text)

위의 메서드를 이용해서 어떠한 요소도 찾지 못하는 경우에 발생하는 예외는
    NoSuchElementException

- DOM 요소에 적용할 수 있는 메서드들 / 속성들
clear()      글자를 입력할 수 있는 요소의 글자를 지운다.
click()          요소를 클릭한다.
get_attribute(name)         요소의 속성 중에 name에 해당되는 속성의 값을 추출한다.
is_displayed()              요소가 화면에 출력되는지 확인
is_enabled()                요소가 활성화 되었는지 확인
is_selected()               (체크박스나 셀렉트박스 등) 요소가 선택 상태인지를 확인
"""