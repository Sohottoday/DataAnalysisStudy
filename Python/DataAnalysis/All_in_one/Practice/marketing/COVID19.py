import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import folium

coronaDF = pd.read_csv('COVID_Data_Basic.csv')
"""
Confirmed : 확진자
Death : 사망자
newConfirmed : 신규확진자
"""

print(coronaDF.head())
print(coronaDF.isnull().sum())
print(coronaDF.info())

# Date가 object로 설정되어 있는 것을 확인할 수 있다. 따라서 datetime으로 바꿔준다.
coronaDF['Date'] = pd.to_datetime(coronaDF['Date'])
print(coronaDF.info())

# 최신 데이터만 남기기
latestDF = coronaDF[coronaDF['Date'] == max(coronaDF['Date'])]

# 국가별 합계 구하기
latest_country_sum = latestDF.groupby('Country')['Confirmed', 'Death', 'Recovered'].sum().reset_index()

# 확진자 높은 국가순으로 정렬
latest_country_sum.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)      # reset_index 의 drop에 True를 주면 기존 인덱스 날림
print(latest_country_sum)

# 데이터 전체 조회
latest_country_sum.style.background.gradient(cmap='Reds')     # colab이나 jupyter같은 곳에서 사용하면 됨.



