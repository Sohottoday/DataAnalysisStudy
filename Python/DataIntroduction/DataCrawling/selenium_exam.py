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
screenshot(filename)        화면을 캡처하여 filename으로 저장한다.
send_keys(value)            키를 입력한 뒤 일반적으로 text 데이터를 보낸다.

value가 텍스트 데이터가 아닌 경우(특수키 : 방향키, 펑션키(f1, f2, f3 ... f12), Enter, Tab, Control ...)
즉, 특수키를 사용해야 하는 경우에는 별도의 모듈을 사용해야 한다.
from selenium.Webdriver.common.keys import Keys

방향키 : ARROW_DOWN / ARROW_LEFT / ARROW_RIGHT / ARROW_UP
BACKSPACE / DELETE / HOME / END / INSERT / ALT / COMMAND / CONTROL / SHIFT / ENTER / ESCAPE / SPACE / TAB
F1 / F2 / F3 / F4 ... / F12

submit()            입력 양식을 전송
value_of_css_property(name)         name에 해당하는 css속성 값을 추출

/////////속성들/////////
id              요소의 id 속성
location        요소의 위치
parent          부모 요소
rect            크기와 위치정보를 가진 딕셔너리 자료형을 리턴한다.
screenshot_as_base64        스크린샷을 base64형태로 추출한다.
screenshot_as_png           스크린샷을 PNG형식의 바이너리로 추출한다.
size            요소의 크기
tag_name        태그 이름
text            요소 내부의 글자


//////// PhantomJS 용 메서드와 속성 ////////
add_cookie(cookie_dict)             쿠키 값을 딕셔너리 형식으로 지정
>> driver.add_cookie({'name':'kim', 'age':'56'})
   driver.add_cookie({'name':'kim', 'value':'test', 'path':'/'})
   driver.add_cookie({'name':'kim', 'value':'test', 'path':'/', 'secure':True})

back() / forward()          이전 페이지 또는 다음 페이지로 이동
close()                     브라우저를 닫다
current_url                 현재 url을 추출한다.
delete_all_cookies()        모든 쿠키를 제거한다.
execute(command, params)    브라우저의 고유 명령어를 실행
get(url)                    웹 페이지를 읽어들인다.
get_screenshot_as_file(filename)        스크린샷을 파일이름으로 저장
get_screenshot_as_png           png형식으로 스크린샷의 바이너리 추출

implicitly_wait(sec)        최대 대기시간을 초 단위로 지정해서 처리가 끝날 때 까지 대기
quit()                      드라이버를 종료시켜서 브라우저를 닫는다.


"""

from selenium import webdriver

url = "http://www.naver.com/"

# PhantomJS 드라이버 추출
driver = webdriver.PhantomJS('F:/phantomjs-2.1.1-windows/bin/phantomjs')

driver.implicitly_wait(3)          # 드라이버를 초기화될 때까지 3초간 대기

driver.get(url)

driver.save_screenshot("naver.png")

driver.quit()



# 로그인 해보기
user = "사용할 아이디"
mypwd = "여러분의 비밀번호"

driver = webdriver.PhantomJS('F:/phantomjs-2.1.1-windows/bin/phantomjs')
driver.implicitly_wait(3)

url_login = "https://nid.naver.com/nidlogin.login"

driver.get(url_login)
print("로그인 페이지를 열었습니다.")

# 아이디를 입력하는 input 요소를 찾아오기
inputID = driver.find_element_id("id")

# 입력박스에 있는 텍스트를 모두 지워준다.
inputID.clear()

# 입력박스에 아이디 입력하기
inputID.send_keys(user)

# 비밀번호를 입력하는 input 요소를 찾아오기
inputPW = driver.find_element_id("pw")
inputPW.clear()

# 비밀번호 입력박스에 비밀번호 입력하기
inputPW.send_keys(mypwd)

# 로그인 버튼을 찾기
loginBin = driver.find_element_by_css_selector("input.btn_global[type=submit]")

# 아이디와 비밀번호를 전송한다.
loginBin.submit()
print('로그인 버튼을 클릭하였습니다.')


# 웹 드라이브를 크롬으로 할 때에는
driver = webdriver.Chrome('F:/chromedriver_win32/chromedriver')
driver.implicitly_wait(3)

driver.get("http://en.wikipedia.org")
driver.execute_script("window.alert('하이!! selenium!!')")






# execute_script() 메서드를 이용한 자바스크립트 처리
driver.implicitly_wait(3)

driver.get("https://google.com")
res = driver.execute_script("return 10+10")
print(res)



