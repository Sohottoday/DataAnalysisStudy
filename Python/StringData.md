# 문자열을 입맛대로 처리하기



- Pandas dtype
  - object : 문자열 타입
  - int64 : 정수 타입
  - float64 : 부동소수점 실수 타입
  - bool : 불리언 타입
  - datetime64 : Datetime 타입
  - timedelta[ns] : 두 datetime 간의 차이
  - category : 카테고리 목적의 문자열



```python
import pandas as pd

df = pd.read_excel('~~~.xlsx')

df['곡명']

df['곡명'].str		# 문자열 함수를 쓰겠다 라는 의미
df['곡명'].str[0]		# 첫번째 문자열을 불러옴, 0부터 쭉 숫자-1번째 문자열을 불라온다는 의미
df['곡명'].str.split()		# 단어를 쪼개어 리스트 형식으로 반환
df['곡명'].str.split(expand = True)		# 단어를 쪼개어 데이터프레임 형식으로 반환
df['곡명'].str.split(expand = True)[0]		# 데이터프레임 형식으로 반환하여 첫번째 값 반환
df['곡명'].str.split(expand = True)[[0, 1]]		# 리스트 방식으로 주어 여러값 반환 가능
df['곡명'].str.startswith('하')		# '하'라고 시작하는 값 True로 반환
cond = df['가수'].str.startswith('임')
df[cond]		# 가수중에 '임'으로 시작하는 가수를 변수에 담아 데이터 프레임형식으로 출력
df['곡명'].str.endsswith		# 끝나는 부분도 지정 가능

cond = df['곡명'].str.contains('사랑')		# 사랑이라는 단어가 포함된 곡명을 반환
df['곡명'].str.count('사랑')		# 사랑이라는 단어가 몇번이나 들어가 있는지 카운트를 반환
df['곡명'].str.strip()		# 문자열 좌우에 있는 공백 모두 제거 **
df['곡명'].str.lstrip()		# 문자열 왼쪽의 공백 제거
df['곡명'].str.rstrip()		# 문자열 오른쪽의 공백 제거
df['곡명'] = df['곡명'].str.lstrip('!')		# 문자열 왼쪽의 느낌표 제거
df['곡명'] = df['곡명'].str.lower()		# 문자열의 영어부분을 모두 소문자로 변경
df['곡명'] = df['곡명'].str.upper()		# 문자열의 영어부분을 모두 대문자로 변경
df['곡명'].str.replace('사랑', '')		# 사랑이란 단어를 모두 공백으로 변경
df['곡명'] = df['곡명'].str.replace('사랑', 'Love')		# 사랑이라는 단어를 모두 Love로 변경
df['곡명_변경'] = df['곡명'].str.replace('사랑', 'Love')		# 사랑이라는 단어를 모두 Love로 변경하여 곡명_변경이라는 컬럼 추가

def fn(val):
    val = val.replace('...')
    return len(val)		# 글자수 리턴

df['곡명_글자수'] = df['곡명'].apply(fn)		# apply를 통해 정의된 함수 수행도 가능



```

