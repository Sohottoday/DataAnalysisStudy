import pandas as pd

# Series
pd.Series([1, 2, 3, 4])

a = [1, 2, 3, 4]
print(pd.Series(a))

# DataFrame
company1 = [['삼성', 2000, '스마트폰'],
            ['현대', 1000, '자동차'],
            ['네이버', 500, '포털']]

df1 = pd.DataFrame(company1)
print(df1)
print("-----------")

## 제목 컬럼 만들기
df1.columns = ['기업명', '매출액', '업종']
print(df1)
print("-----------")

## dict로 DataFrame 만들기
company2 = {'기업명' : ['삼성', '현대', '네이버'],
            '매출액' : [2000, 1000, 500],
            '업종' : ['스마트폰', '자동차', '포털']}

df2 = pd.DataFrame(company2)
print(df2)
print("-----------")

## index를 특정 column으로 지정하기
df1.index = df1['기업명']
print(df1)
print("-----------")

# 구글 드라이브에 있는 샘플 파일 읽어오기 (colab에서만 사용 가능)
"""
1. 로컬에서 파일 읽어오기(초심자가 진행하기에는 번거로움)
from google.colab import files
myfile = files.upload()

import io
pd.read_csv(io.BytesIO(myfile['korean-idol.csv']))

2. 구글 드라이브에 있는 샘플 파일 읽어오기
from google.colab import drive
drive.mount('/content/drive')

filename = '이곳에 파일 경로 붙여넣기'
pd.read_csv(filename)

3. 파일 URL로부터 바로 읽어오기
pd.read_csv('http://bit.ly/ds-korean-idol')

"""

# 엑셀 파일 읽어오기
"""
from google.colab import drive
drive.mount('/content/drive')
filename = '파일 경로 붙여넣기'

pd.read_excel(filename)
"""

df = pd.read_csv('korean-idol.csv')
print(df)

# 열 (column) 출력하기
print(df.columns)

# 열 이름 재정의하기
new_col = ['name', '그룹', '소속사', '성별', '생년월일', '키', '혈액형', '브랜드평판지수']
df.columns = new_col

# index(행) 출력하기
print(df.index)

# info는 기본적인 행(row)의 정보와 데이터 타입을 알려준다.
# info 메서드는 주로 빠진 값(null 값)과 데이터 타입을 볼 때 활용한다.
print(df.info())

# 통계 정보 알아보기(describe)
print(df.describe())

# 형태 알아보기(shape)
print(df.shape)

# 오름차순 정렬
print(df.sort_index())

# 내림차순 정렬
print(df.sort_index(ascending=False))

# column 별로 정렬
print(df.sort_values(by='키'))      # 키 컬럼의 순서를 오름차순으로 정렬(by 없이도 가능하다.)
print(df.sort_values('키', ascending=False))

# 복수 정렬
print(df.sort_values(by=['키', '브랜드평판지수']))           # 키가 동일하다면 브랜드 평판 지수로 그 다음 정렬한다.

# column을 선택하는 방법
print(df['name'])
print(df.키)        # 이 방법은 코드의 가독성 때문에 권장하지 않음


