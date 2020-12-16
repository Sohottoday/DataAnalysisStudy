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

