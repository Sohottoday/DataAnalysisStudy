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

# read_csv()    parse_dates 에 날짜 형식의 column 을 넣어주면 자동적으로 datetime 형태로 변환시켜준다.

df_order = pd.read_csv('olist_orders_dataset.csv', parse_dates=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
print(df_order.shape)
print(df_order.columns)
print(df_order.sample(3))
print(df_order.info())
print(df_order.dtypes)

df_order_item = pd.read_csv('olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])
df_order_review = pd.read_csv('olist_order_reviews_dataset.csv', parse_dates=['review_creation_date', 'review_answer_timestamp'])
df_order_pay = pd.read_csv('olist_order_payments_dataset.csv')
df_cust = pd.read_csv('olist_customers_dataset.csv')
df_product = pd.read_csv('olist_products_dataset.csv')
df_seller = pd.read_csv('olist_sellers_dataset.csv')
df_category = pd.read_csv('product_category_name_translation.csv')
df_geo = pd.read_csv('olist_geolocation_dataset.csv')

print(df_order.info())

