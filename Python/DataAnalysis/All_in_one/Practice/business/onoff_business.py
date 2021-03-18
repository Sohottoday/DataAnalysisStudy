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

df_order = pd.read_csv('olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])


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


# olist_orders_dataset 테이블에 새로운 정보 추가
"""
    - order_purchase_timestamp : 구매 시작 날짜/시간
    - order_approved_at : 결제 완료 날짜/시간
    - order_delivered_customer_date : 실제 고객한테 배달완료된 날짜/시간
    - order_estimated_delivery_date : 시스템에서 고객에게 표시되는 예상배달날짜
    - order_approved_at - order_purchase_timestamp : pay_read_time(단위: 분)
    - order_delivered_customer_date - order_approved_at : delivery_lead_time(단위: 일)
    - order_estimated_delivery_date - order_delivered_customer_date : estimated_date_miss(단위: 일)
"""
print('-' * 30)
print(df_order_clean.info())

# 결제리드타임
df_order_clean['pay_lead_time'] = df_order_clean['order_approved_at'] - df_order_clean['order_purchase_timestamp']
print(df_order_clean['pay_lead_time'])

# 추후 계산을 용이하게 하기 위해 분 단위로 변경(pandas의 timedelta)
df_order_clean['pay_lead_time_m'] = df_order_clean['pay_lead_time'].astype('timedelta64[m]')
print(df_order_clean['pay_lead_time_m'])

# 배달 리드타임 - 일 단위
df_order_clean['delivery_lead_time'] = df_order_clean['order_delivered_customer_date'] - df_order_clean['order_approved_at']
df_order_clean['delivery_lead_time_D'] = df_order_clean['delivery_lead_time'].astype('timedelta64[D]')
df_order_clean['delivery_lead_time_D']

# 예상날짜 틀린정도 - 일 단위
df_order_clean['estimated_date_miss'] = df_order_clean['order_estimated_delivery_date'] - df_order_clean['order_delivered_customer_date']
df_order_clean['estimated_date_miss_D'] = df_order_clean['estimated_date_miss'].astype('timedelta64[D]')
df_order_clean['estimated_date_miss_D']

# 세 컬럼 모두 정수로 바꿔준다.
df_order_clean['pay_lead_time_m'] = df_order_clean['pay_lead_time_m'].astype(int)
df_order_clean['delivery_lead_time_D'] = df_order_clean['delivery_lead_time_D'].astype(int)
df_order_clean['estimated_date_miss_D'] = df_order_clean['estimated_date_miss_D'].astype(int)

# 히스토그램 출력

plt.subplot(1, 3, 1)
#plt.figure(figsize=(12, 6))
sns.distplot(df_order_clean['pay_lead_time_m'])
plt.title('pay_lead_time(분)')

plt.subplot(1, 3, 2)
sns.distplot(df_order_clean['delivery_lead_time_D'])
plt.title('delivery_lead_time_D')

plt.subplot(1, 3, 3)
sns.distplot(df_order_clean['estimated_date_miss'])
plt.title('estimated_date_miss_D')

plt.show()

"""
- pay_lead_time의 경우, 오른쪽으로 치우친 형태를 띄고 있는 것으로 볼 때, 다소 큰 양(+)의 값들이 포진해있다고 생각해 볼 수 있다.
- delivery_lead_time의 경우, 오른쪽으로 치우친 형태를 띄고 있는 것으로 볼 때, 다소 큰 양(+)의 값들이 포진해있다고 생각해볼 수 있다.
- estimated_date_miss의 경우, 양(+)의 값들과 음(-)의 값들이 모두 포진해 있다.
"""

# 새로 추가한 컬럼들의 요약 정보
print(df_order_clean[['pay_lead_time_m', 'delivery_lead_time_D', 'estimated_date_miss_D']].describe())
## pay_lead_time의 경우, 중앙값은 20인데, 평균값을 616이므로 매우 큰 수치의 값들이 함께 있다고 생각할 수 있다.
## 이 외에도 수치를 보면 잘못된 정보들이 있다는 것을 알 수 있다.

# 이상한 데이터 확인
# print(df_order_clean[df_order_clean['pay_lead_time_m']==44486])
# print(df_order_clean[df_order_clean['delivery_lead_time_D']==208])
# print(df_order_clean[df_order_clean['destimated_date_miss_D']==146])
print(df_order_clean[df_order_clean['delivery_lead_time_D']==-7])

# 이러한 이상한 정보들을 더 살펴보기 위한 boxplot을 그려본다.

# boxplot을 위한 input format
df_order_time = df_order_clean[['pay_lead_time_m', 'delivery_lead_time_D', 'estimated_date_miss_D']]

# boxplot 출력
## 분단위, 일단위 등으로 다르게 설정되어 있기 때문에 하나씩 봐야한다.

plt.subplot(1, 3, 1)
sns.boxplot(data=df_order_time['pay_lead_time_m'], color='red')

plt.subplot(1, 3, 2)
sns.boxplot(data=df_order_time['delivery_lead_time_D'], color='blue')

plt.subplot(1, 3, 3)
sns.boxplot(data=df_order_time['estimated_date_miss_D'], color='green')

plt.show()

# 이상치(outlier) 검출
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])      # percentile : 값을 퍼센트로 표시해 주는 함수.
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)

    return np.where((data > upper_bound) | (data < lower_bound))        # where : 배열의 요소값이 특정 조건에 만족하는 값을 반환하는 함수, 인덱스(위치) 값을 반환한다.

# 이상치 수
print(outliers_iqr(df_order_time['pay_lead_time_m']))
print(outliers_iqr(df_order_time['pay_lead_time_m'])[0].shape[0])
print(outliers_iqr(df_order_time['delivery_lead_time_D'])[0].shape[0])
print(outliers_iqr(df_order_time['estimated_date_miss_D'])[0].shape[0])

# 세 컬럼의 이상치 row 인덱스 출력
pay_lead_outlier_index = outliers_iqr(df_order_time['pay_lead_time_m'])[0]
del_lead_outlier_index = outliers_iqr(df_order_time['delivery_lead_time_D'])[0]
est_lead_outlier_index = outliers_iqr(df_order_time['estimated_date_miss_D'])[0]

# 이상치에 해당되는 값 출력
print(df_order_time.loc[pay_lead_outlier_index, 'pay_lead_time_m'])

# 이상치 제거
# numpy concat을 통한 array 배열 합치기
lead_outlier_index = np.concatenate((pay_lead_outlier_index, del_lead_outlier_index, est_lead_outlier_index), axis=None)

# for문을 이용해 이상치가 아닌 리드타임 값의 인덱스를 추려준다.
lead_not_outlier_index = []

for i in df_order_time.index:
    # lead_outlier_index에 포함되지 않는다면 추가
    if i not in lead_outlier_index:
        lead_not_outlier_index.append(i)

df_order_time_clean = df_order_time.loc[lead_not_outlier_index]
df_order_time_clean = df_order_time_clean.reset_index(drop=True)

# 클렌징한 df의 요약정보
print(df_order_time_clean.describe())


# olist_customer_dataset 데이터 살펴보기(상품을 주문한 고객들은 어떤 고객들인지)
print(df_cust.info())

# states별 고객 수
cust_stat = pd.DataFrame(df_cust['customer_state'].value_counts()).reset_index()
cust_stat.columns = ['states', 'people_lives']

# states별 고객 수 시각화
sns.barplot(x='states', y='people_lives', data=cust_stat)
plt.show()

# 도시 별 고객 비율(퍼센트)
df_cust['customer_city'].value_counts(normalize=True) * 100       # normalize 를 True로 주면 비율로 나타내준다. 100을 곱해주는 이유는 퍼센트로 나타내기 위함.

# 상위 10개 출력
df_customer_city = pd.DataFrame(df_cust['customer_city'].value_counts(normalize=True) * 100).reset_index()
df_customer_city.columns = ['city', 'people_lives_perc']

print(df_customer_city.loc[:10, :])

# pie chart로 나타내보기
labels = df_customer_city['city'].values[:10]
sizes = df_customer_city['people_lives_perc'].values[:10]

explode = (0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=30, textprops={'fontsize':16})
ax1.axis('equal')

plt.tight_layout()
plt.title('도시별 고객이 살고 있는 비율', fontsize=20)
plt.show()


# olist_order_items_dataset : 실제 주문한 상품들 데이터 확인
print(df_order_item.info())

# 결측치 존재 확인
print(df_order_item.isnull().sum())

# item_id를 제일 많이 갖고 있는 order_id 출력
temp = pd.DataFrame(df_order_item.groupby(by=['order_id'])['order_item_id'].count().reset_index())
temp.columns = ['order_id', 'order_item 수']
temp[temp['order_item 수'] == temp['order_item 수'].max()]

"""
# 위 결과로 알 수 있는 점
- order_item_id 컬럼의 뜻은 하나의 주문(order_id)에서 구매한 상품들의 수르 뜻한다.(종류 상관 x)
- 위 결과의 order_item_id 1번인 상품과 2번, 12번 상품의 price는 1.2로 같지만, 21번 상품은 7.8로 다른 것을 볼 때, price 컬럼은 '상품 단가'를 의미한다고 생각해 볼 수 있다.(위 결과에서 총 삼품 종류는 3가지)
- 하나의 주문번호에서 '상품별 매출액'을 산출해내기 위해서는 각 상품들의 '구매수량' 컬럼을 추가해줘야 한다.(상품단가 * 구매수량 = 상품별 매출액)
"""

# 주문한 상품 수량 : order_prod_quantity 컬럼을 만든다.
df_qt = pd.DataFrame(df_order_item.groupby(by=['order_id', 'product_id'])['order_item_id'].count().reset_index())

# 컬럼 명 변경
df_qt.columns = ['order_id', 'product_id', 'order_prod_quantity']


# 상품 별 주문수량을 추가해주기 위한 merge

df_order_item_col = ['order_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value']

# merge
df_order_item = pd.merge(df_order_item[df_order_item_col], df_qt, how='inner', on=['order_id', 'product_id'])

# 컬럼 순서 재배치
df_order_item = df_order_item[['order_id', 'product_id', 'price', 'freight_value', 'order_prod_quantity', 'shipping_limit_date', 'seller_id']]
# 이럴 경우 조인할 때 데이터프레임의 shape를 그대로 가져오기 때문에 row 중복이 발생한다.
## 발생됨 row 중복 제거 방법 : drop_duplicates

df_order_item.drop_duplicates(inplace=True)
df_order_item.reset_index(drop=True, inplace=True)


# 주문(order_id)내 상품 별 매출액 계산
df_order_item['order_amount'] = df_order_item['price'] * df_order_item['order_prod_quantity']
df_order_item.reset_index(drop=True, inplace=True)

# 실제 어떤 상품들이 구매되었는지 확인
"""
# 실제 주문 내역 테이블을 기준으로 상품 정보 매칭
- olist_order_items_dataset
- olist_products_dataset
- product_category_name_translation
"""

# columns 확인
print(df_product.columns)

print(df_product.isnull().sum())
## product_category_name이 포르투갈 어로 적혀있기 때문에 영어 명칭 테이블과 매칭시켜 준다.
## merge를 활용하여 병합

df_product_cat = pd.merge(df_product, df_category, how='left', on=['product_category_name'])

# 컬럼 순서 재배치
df_product_cat = df_product_cat[['product_id', 'product_category_name', 'product_category_name_english', 'product_name_lenght', 'product_description_lenght',
'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']]

# 다시 결측치 확인
print(df_product_cat.isnull().sum())


# df_order_item을 기준으로 merge
df_order_item_prod = pd.merge(df_order_item, df_product_cat, how='left', on=['product_id'])

print(df_order_item_prod.describe())
print(df_order_item_prod.isnull().sum())

# 모든 컬럼을 기준으로 null값이 존재하는 row 출력
print(df_order_item_prod[df_order_item_prod.isnull().any(axis=1)])

# 모든 결측치를 포함한 row를 drop한다.
df_order_item_prod_clean = df_order_item_prod.dropna(axis=0)
df_order_item_prod_clean.reset_index(drop=True, inplace=True)

print(df_order_item_prod_clean.shape)
print(df_order_item_prod_clean.isnull().sum())

# 상품 카테고리 별 정보를 살펴보자

"""
전체 평균값과 비교하며 카테고리별로 살펴본다.
- 상품 카테고리 별 평균 제목/설명 길이, 사진 수
- 상품 카테고리 별 상품의 평균 무게/길이/높이/너비
"""

# pandas 소수점 출력 설정 : 소수점 2째자리
pd.options.display.float_format = '{:.2f}'.format

# 상위 10개만 확인
cat_top10 = df_order_item_prod_clean['product_category_name_english'].value_counts()[:10]

# 상품 카테고리가 cat_top10의 값에 포함되는 row만 출력하기 (pandas의 isin 사용)
## isin : 조건에 해당되는 값을 반환

df_cat_10 = df_order_item_prod_clean[df_order_item_prod_clean['product_category_name_english'].isin(cat_top10.index)].reset_index(drop=True)

# 상위 10개의 카테고리별 정보 - 상품등록정보
cat_info1_col = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']

print(df_cat_10.groupby('product_category_name_english')[cat_info1_col].mean())

print(df_cat_10[cat_info1_col].describe())

# 카테고리별 정보2 확인
cat_info2_col = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
print(df_cat_10.groupby('product_category_name_english')[cat_info2_col].mean())
print(df_cat_10[cat_info2_col].describe())


# 2016년부터 2018년까지 어떤 상품 카테고리가 많은 매출을 기록하고 있는지 확인해보자

# 상품 카테고리 종류 수
print('상품 카테고리 종류 수 : {} 종류'.format(len(df_order_item_prod_clean['product_category_name_english'].unique())))

# 매출액 기준 상품 카테고리
temp = pd.DataFrame(df_order_item_prod_clean.groupby(by=['product_category_name_english'])['order_amount'].sum())

# 매출액 높은 순으로 정렬
temp = temp.sort_values(by='order_amount', ascending=False)
print(temp)


# squarify 를 활용해 트리맵으로 나타내기
import squarify

plt.figure(figsize=(12, 10))
squarify.plot(sizes=temp['order_amount'][:10], label=temp.index.values[:10], alpha=.7)
plt.show()
## 가장 높은 매출액을 보이는 상품 카테고리는 'health_beauty'이다.


# 상품 카테고리별 주문 수와 매출액의 관계 알아보기
# 카테고리별 주문수 확인
df_cat_order_cnt = pd.DataFrame(df_order_item_prod_clean['product_category_name_english'].value_counts())
df_cat_order_cnt = df_cat_order_cnt.reset_index()
df_cat_order_cnt.columns = ['category', 'order_cnt']
print(df_cat_order_cnt)

# 카테고리별 매출액 순
df_cat_amount = pd.DataFrame(df_order_item_prod_clean.groupby(by=['product_category_name_english'])['order_amount'].sum())
df_cat_amount = df_cat_amount.sort_values(by='order_amount',ascending=False)
df_cat_amount = df_cat_amount.reset_index()
df_cat_amount.columns = ['category', 'order_amount']
print(df_cat_amount)

# 두 테이블을 비교해주기 위해 스케일을 맞춰준다.
# 카테고리별 주문 비율

df_cat_order_cnt['order_cnt_perc'] = (df_cat_order_cnt['order_cnt'] / sum(df_cat_order_cnt['order_cnt']))*100

# 카테고리별 매출 비율

df_cat_amount['order_amount_perc'] = (df_cat_amount['order_amount'] / sum(df_cat_amount['order_amount']))*100

# 카테고리별 주문수와 매출액 테이블 결합

df_cat = pd.merge(df_cat_order_cnt, df_cat_amount,
        how='inner', on='category')

# 상위 10개 추출
df_cat = df_cat.sort_values(by='order_amount', ascending=False)
df_cat = df_cat.reset_index(drop=True)
print(df_cat)

# pandas의 melt 를 활용하여 x축과 y축에 들어갈 수 있는 데이터로 변환
df_cat_melt = pd.melt(df_cat[:10], id_vars=['category'], value_vars=['order_cnt_perc', 'order_amount_perc'])
print(df_cat_melt)

# barplot으로 시각화
plt.figure(figsize=(12, 10))
ax = sns.barplot(data=df_cat_melt, x='category', y='value', hue='variable', color='salmon')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()

"""
매출액이 가장 높은 health_beauty의 경우, 주문수도 함께 많은 것을 알 수 있습니다. 
watches_gifts는 주문수는 상대적으로 많은 편이 아니지만, 전체 매출에서 두번째로 높은 모습을 보이고 있는 것으로 볼 때, 
시계와 같이 객당 단가가 높은 상품이 포함된 카테고리의 특징으로 생각해볼 수 있습니다.
"""