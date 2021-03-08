import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
# latest_country_sum.style.background.gradient(cmap='Reds')     # colab이나 jupyter같은 곳에서 사용하면 됨.


# 날짜별 확진자, 사망자, 회복자 합계 구하기
date_status = coronaDF.groupby('Date')['Confirmed', 'Death', 'Recovered'].sum().sort_index()
print(date_status)

# 시간에 따른 누적 확진자, 사망자, 회복자 그래프(seaborn lineplot)
plt.figure(figsize=(18, 8))
plt.xticks(rotation=45)
sns.lineplot(data=date_status)
plt.show()

# folium이란?
"""
지도 데이터에 'Leaflet.js'를 이용하여 위치정보 시각화 라이브러리
python-visualization.github.io/folium
"""

# folium 사용해보기
m = folium.Map()
m.save('map.html')      # vscode에서 사용하기 때문에 이와같이 변수에 넣고 html로 변환하여 보는 것이다. jupyter나 colab 사용시에는 그냥 folium.Map() 만 해도 바로 적용된다.

# folium으로 서울 지도 표시
folium.Map(location=[37.564837, 126.980343], zoom_start=16)

# CircleMarker 그리기, 색상 채우기
# CircleMarker로 popup 표시
m = folium.Map(location=[37.564837, 126.980343], zoom_start=16)
folium.CircleMarker([37.564837, 126.980343], radius=100, color='red', fill=True, fill_color='red', popup='Seoul').add_to(m)        # 서클 마커를 추가한다는 의미, radius 는 반지름을 의미한다.

m.save('seoul.html')


# 국가별 최신 확진자 데이터와 Folium을 활용하여 확진자 표시해보기
# 확진자 수가 많으면 반지름이 크게 표시 ,popup은 확진자 수 표시
"""
m = folium.Map(location=[0, 0], zoom_start=2)
for n in latestDF.index:
    folium.CircleMarker([latestDF['Lat'][n], latestDF['Long'][n]], radius=int(latestDF['Confirmed'][n]/1000), color='red', fill=True, fill_color='red', popup=latestDF['Country'][n] + " : " + str(latestDF['Confirmed'][n])).add_to(m)

m.save('COVID19.html')

이유는 알 수 없으나 강의 데이터와 다르게 해당 kaggle 데이터는 국가의 위도경도를 표시하지 않음.
"""
