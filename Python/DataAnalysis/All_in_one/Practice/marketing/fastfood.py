import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno
import folium
import numpy as np
import glob

import matplotlib.font_manager as fm

path = 'C:\\Users\\user\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumSquareRoundEB.ttf'
font_name = fm.FontProperties(fname=path, size=14).get_name()
print(font_name)
plt.rc('font', family=font_name)

# 설정한 폰트는 rebuild 해줘야 적용된다.
fm._rebuild()

df_store = pd.read_csv('C:\\Users\\user\\Desktop\\Programing\\Github\\SelfStudy\\Python\\DataAnalysis\\All_in_one\\Practice\\marketing\\소상공인시장진흥공단_상가(상권)정보_제주_202012.csv', encoding='utf-8', delimiter='|')

print(df_store.shape)
print(df_store.columns)

print(df_store.info())
print(df_store.isnull().sum())

plt.figure(figsize=(16, 6))
sns.heatmap(df_store.isnull(), cbar=False)
plt.xticks(rotation=45, fontsize=12)
plt.show()

msno.matrix(df_store, fontsize=12, figsize=(16, 6))
plt.show()

# 결측치 처리하기
"""
결측치는 삭제 혹은 특정값으로 채우는 방법이 있다.
데이터가 많은 경우 데이터를 삭제할 수도 있겠지만, 데이터는 소중하기 때문에 특정값으로 대체하게 된다.
pandas
    fillna
    dropna
    notnull
    pandas.DataFrame.any
"""

# 여기서는 결측값이 비교적 적은 컬럼들만 선정해서 진행
df_clr_columns = ['상가업소번호', '상호명', '상권업종대분류코드', '상권업종대분류명', '상권업종중분류코드',
       '상권업종중분류명', '상권업종소분류코드', '상권업종소분류명', '시도코드',
       '시도명', '시군구코드', '시군구명', '행정동코드', '행정동명', '법정동코드', '법정동명', '지번코드',
       '대지구분코드', '대지구분명', '지번본번지', '지번주소', '도로명코드', '도로명', '건물본번지',
       '도로명주소', '신우편번호', '경도', '위도']

df_store_clr = df_store[df_clr_columns].copy()

print(df_store_clr.shape)
print(df_store_clr.isnull().sum())

# null값이 있는 row 제거
df_store_clean = df_store_clr.dropna(axis=0)


# 지리정보(위도, 경도)를 이용해 지도 그려보기
df_store_clean.plot.scatter(x='경도', y='위도', figsize=(12, 8), grid=True)
plt.show()

print(df_store_clean['시군구명'][:10])

# '시군구명'을 기준으로 제주시와 그 외 지역으로 구분
df_store_jeju = df_store_clean.loc[df_store_clean['시군구명'].str.startswith('제주시')]
df_store_else = df_store_clean.loc[~df_store_clean['시군구명'].str.startswith('제주시')]
"""
(
ggplot(df_store_jeju)
+ aes(x='경도', y='위도')
+ geom_point(color='green', alpha=0.3, size=0.2)
+ theme(text=element_text(family='NanumSquareRound'))
)
"""

