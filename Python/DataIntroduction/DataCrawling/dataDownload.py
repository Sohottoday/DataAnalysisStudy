# 웹상의 정보를 추출하기 위한 라이브러리 불러오기
# urllib 라이브러리 : http나 FTP를 사용해서 데이터를 다운로드 할 때 사용하는 라이브러리
# urllib : URL을 다루는 모듈을 모아 놓은 패키지
# urllib 패키지에 있는 모듈 중에서 request 모듈을 이용하는데 request 모듈 안에 urlretrieve() 함수를 사용한다.

import urllib.request

url = "https://t1.daumcdn.net/daumtop_chanel/op/20170315064553027.png"
imgName = 'C:/Users/user/Desktop/Programing/Github/SelfStudy/Python/DataIntroduction/DataCrawling/daum.png'

urllib.request.urlretrieve(url, imgName)        # urlretrieve(URL, 저장할 파일 경로)
print('다운로드가 완료 되었습니다.')


# request.urlopen() 함수 사용하기
## urlretrieve() 함수는 파일로 곧바로 저장하지만 urlopen() 함수는 파일로 바로 저장하지 않고 메모리에 로딩을 한다.

url = "https://t1.daumcdn.net/daumtop_chanel/op/20170315064553027.png"
imgPath = "C:/Users/user/Desktop/Programing/Github/SelfStudy/Python/DataIntroduction/DataCrawling/daum2.png"

downImg = urllib.request.urlopen(url).read()

# 파일로 저장하는 과정
# f = open("a.txt", ,"w")
# f.write("테스트로 파일에 내용을 적습니다")
# f.close()

# 위의 과정을 with 문으로 간단하게 표현한다.
# with.open("a.txt", "w") as f:
# f.write("테스트로 파일에 내용을 적습니다.")

with open(imgPath, mode="wb") as f:      # w는 쓰기모드, b는 바이너리 모드(이미지)
    f.write(downImg)

print("이미지 다운로드 완료")



