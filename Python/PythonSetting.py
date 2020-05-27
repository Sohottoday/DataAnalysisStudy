# Anaconda Python : 과학계산에 특화된 파이썬 배포판
# 공식 배포판 파이썬을 제거한 뒤 아나콘다 파이썬을 설치해야 한다.
# 환경변수 옵션 체크
# Anaconda -> Individual Edition -> download -> 자신의 버전에 맞는 것 다운로드
# All user 체크 -> 왠만해선 C드라이브 바로 아래 설치 -> 환경변수를 설정하겠다는것에 꼭 체크한 뒤 설치
#

# Jupyter Notebook : 웹 브라우저를 통해 파이썬을 실행하고 다양한 기능 제공
# window powershell : 작업 환경
# python : 기본 파이썬 환경
# ipython : 파이썬보다 조금 더 다양한 기능 제공
# 파워 쉘의 한계 = 글자만 출력 가능
# Jupyter Notebook : 웹브라우저로 출력하기 때문에 그래프나 표같은것들도 출력 가능

# Jupyter Notebook 데이터 분석
# 1. 접속 암호 설정
# 2. Jupyter Notebook 서버 실행
# 3. http://localhost:8888 주소로 접속하여 설정한 암호 입력
# 4. 새로운 Python 노트북을 생성하면 노트북 안에서 Cell 단위로 코드 실행

# jupyter --version : 버전 확인
# jupyter notebook password : 암호 입력(커서가 움직이지 않는다.)
# jupyter notebook : 주피터 노트북 서버 가동(활용하는 동안에는 종료하면 안된다.)
# Ctrl + c 로 서버 종료시킬 수 있음
# 파워쉘의 경로 기준으로 폴더가 실행된다.

# 메인 화면 오른쪽 위의 new 에서 python3 를 클릭하여 파일 생성 후 작업
# untitled 를 클릭하여 파일 이름 변경
import pasdas as pd

df = pd.read_excel('data/환율-20180914.xls', index_col='통화명')
df.head()
# 위 부분을 작성하여 나오는 결과값 확인해 볼 것 => 표 형식으로 결과값 출력됨

# 한글 폰트 설정
%matplotlib inline
from matplotlib import rc
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
rc('font', family= 'Malgun Gothic')  #window일 경우
#rc('font',family='AppleGothic') Mac일 경우

df2 = df.sort_values('송금 - 보내실 때', ascending = True)    # 지정 빌드 오름차순 정렬

# 지정 2개 컬럼에 대한 가로형 막대그래프를 10인치 * 10인치 크기로 그린다.
df2[['송금 - 보내실 때', '송금 - 받으실 때']].plot(kind='barh', figsize = (10, 10))