# 브라질 온라인 쇼핑몰 Olist 관련 Data

"""
# 학습 목표
1. 이커머스 데이터를 이해하고 이를 위해 데이터셋을 원하는 형태로 바꾸고 적합한 시각화를 진행한다.
2. 데이터 핸들링 시 엑셀의 기능과 비교해서 이해해본다.
3. (참고) 자료를 확인하며 개념과 기능을 숙지한다.
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import squarify     # treemap

import matplotlib.font_manager as fm

sys_font = fm.findSystemFonts()
print(f'sys_font number : {len(sys_font)}')
# print(sys_font)

# 나눔폰트의 수 출력
nanum_font = [f for f in sys_font if 'Nanum' in f]
print(f'nanum_font number : {len(nanum_font)}')
print(nanum_font)

# 설치된 나눔글꼴 중 원하는 글꼴의 전체 경로를 가져온다.
path = 'C:\\Users\\user\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumSquareRoundEB.ttf'
font_name = fm.FontProperties(fname=path, size=14).get_name()
print(font_name)
plt.rc('font', family=font_name)

# fm._rebuild() 를 해주어야 적용된다.
fm._rebuild()

# 코드 전체 폰트 설정
plt.rcParams['font.family'] = 'NanumSquareRound'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

# 만약 matplotlib에서 마이너스 부호가 깨준다면 사용해준다.
mpl.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np

# read_csv()    parse_dates 에 날짜 형식의 column 을 넣어주면 자동적으로 datetime 형태로 변환시켜준다.

df_order = pd.read_csv('olist_orders_dataset.csv', parse_dates=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])


df_order_item = pd.read_csv('olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])
df_order_review = pd.read_csv('olist_order_reviews_dataset.csv', parse_dates=['review_creation_date', 'review_answer_timestamp'])
df_order_pay = pd.read_csv('olist_order_payments_dataset.csv')
df_cust = pd.read_csv('olist_customers_dataset.csv')
df_product = pd.read_csv('olist_products_dataset.csv')
df_seller = pd.read_csv('olist_sellers_dataset.csv')
df_category = pd.read_csv('product_category_name_translation.csv')
df_geo = pd.read_csv('olist_geolocation_dataset.csv')


"""
# 진행 순서
    1. 로드한 테이블을 하나씩 살펴보면서 포함된 데이터를 이해해본다.
    2. 필요한 데이터 핸들링 및 시각화 작업을 진행한다.
    3. 이를 토대로 의미를 도출해본다.
EDA : Exploratory Data Analysis
"""

# Olist_orders_dataset

print(df_order.shape)
print(df_order.columns)
print(df_order.sample(3))
print(df_order.info())
print(df_order.dtypes)
print(df_order.describe(exclude=[np.object]))       # object 타입은 제외하고 보여달라는 의미

# 결측치 수 확인
print('결측치 수 : ', df_order.isnull().sum())

# 결측치가 있는 데이터들의 row 출력
df_order_null = df_order[df_order.isnull().any(axis=1)]     # 널값이 하나라도 있는 row 출력
print(df_order_null)

# df_order에서 order_approved_at 컬럼에 null값이 있는 row만 출력
df_order_col1 = df_order[df_order['order_approved_at'].isnull()]
print(df_order_col1)

# 결측치 시각화
sns.heatmap(df_order.isnull(), cbar=False)
plt.show()

msno.matrix(df_order, fontsize=12, figsize=(12, 6))

"""
# 결측치 처리하기
    - 결측치를 처리하는 방법에는 크게 삭제 혹은 특정값으로 채우는 방법이 있다.
    - 데이터가 많을 경우 데이터를 삭제할 수도 있겠지만, 그렇지 않다면 데이터는 소중하기 때문에 특정값으로 대체하게 된다.
    - 결측치 처리방법은 fillna, dropna, notnull, DataFrame.any 등이 있다.
"""

# 결측치 처리
df_order_clean = df_order.dropna(axis=0)

# 인덱싱 재지정
df_order_clean.reset_index(drop=True, inplace=True)        # 결측치 처리 후 reset_index만 해주면 바로 처리되지 않는다.다. inplace=True 옵션을 줘야한다. drop=True 옵션을 주지 않으면 column에 index라고 표기된 값이 추가된다.

# 주문 상태의 고유값 확인
print('고유한 값 : ', df_order_clean['order_status'].unique())      # 배송 완료와 취소 2가지 존재

# 어떤 주문 상태가 제일 많을까?
print(df_order_clean['order_status'].value_counts())

# null 값이 있는 DF의 주문 상태 고유값
print(df_order_null['order_status'].unique())

# null값이 있는 DF의 주문 상태 별 수
print(df_order_null['order_status'].value_counts())

## 위의 정보들을 볼 때, 고객에게 배송이 완료된 경우에는 취소가 6건 정도지만, 고객에게 결제/배송 진행 도중에 취소된 건수는 619건임을 생각해 볼 수 있다.

"""
# 데이터 시각화의 필요성
    - 표 형태의 데이터에는 많은 정보들이 담겨있지만, 한 눈에 보기 어려우며 요약된 정보만으로는 데이터셋의 특징을 정확히 알기 힘들다.
    - Anscombe's quartet'은 데이터 시각화의 중요성을 이야기할 때 자주 등장하는 예시로, 요약된 정보 만으로 정확한 데이터를 판단할 수 없음을 보여준다.
    - Anscombe's quartet 그래프는 각각이 서로 다른 모양의 그래프를 가졌지만, 4개의 데이터 모두 평균, 표준편차, 상관계수가 같다.

# 시각화 라이브러리
matplotlib
seaborn
plotnine        # R의 ggplot2 와 비슷
plotnine gallery
pandas의 visualization
python graph gallery
"""

# 취소된 건에 대한 차트 비교를 위해 알맞은 데이터 형태로 바꿔준다.
A = df_order_clean[df_order_clean['order_status']=='canceled'].shape[0]
B = df_order_null[df_order_null['order_status']=='canceled'].shape[0]

temp = pd.DataFrame(columns=['del_finished', 'del_not_finished'], index=['cancel_cnt'])
temp.loc['cancel_cnt', 'del_finished'] = A
temp.loc['cancel_cnt', 'del_not_finished'] = B
print(temp)     # del_finished : 배송이 끝난 후 취소된 값, del_not_finished : 배송이 끝나기 전 취소된 값

# pandas의 transpose 로 표의 축 변환
# transpose : x가 order_status, y가 cnt
print(temp.T)

# 취소된 건에 대한 차트 비교 : 세로 막대그래프
plt.figure(figsize=(12, 6))
temp.T.plot(kind='bar')
plt.show()

# 요약 정보
print(df_order_clean.describe(exclude=[np.object]))