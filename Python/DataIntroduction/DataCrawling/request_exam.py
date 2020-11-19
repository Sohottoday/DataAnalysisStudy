# requests 모듈의 메서드
# http에서 사용하는 데이터 전송방식 GET, POST 방식이 있는데, 두 방식의 메서드를 제공

import requests

r = requests.get("http://google.com")            # get 방식의 요청을 하는 경우

# POST 요청
formData = {"key1":"value1", "key2":"value2"}
r = requests.post("http://sample.com", data=formData)


resData = requests.get("http://api.aoikujira.com/time/get.php")

# 텍스트 형식으로 추출
txt = resData.text
print(txt)

# 바이너리 형식으로 데이터 추출
bina = resData.content
print(bina)

# 이미지 데이터 가져오기
res = requests.get('https://t1.daumcdn.net/daumtop_chanel/op/20170315064553027.png')

# 바이너리 형식으로 이미지 저장하기
with open('logo.png', 'wb') as f:
    f.write(res.content)

print("이미지파일이 저장 되었습니다.")



# 세션을 사용하는 경우
session = requests.session()        # 세션 시작하기

# 로그인 하기
login_info = {
    "id" : "userId",
    "passwd" : "userPw"
}

url = "http://www.test.com/loginConfirm.php"        # id와 pw를 확인하는 페이지
result = session.post(url, data=login_info)
result.raise_for_status() # 오류 체크 : 오류가 발생하면 예외처리를 한다.

# 로그인 후 get 방식의 서비스를 요청하는 경우
myUrl = "http://www.test.com/myPage.html"
res = session.get(myUrl)
res.raise_for_status()

from bs4 import BeautifulSoup
soup = BeautifulSoup(res.text, "html.parser")