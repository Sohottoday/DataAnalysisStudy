import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# 통신사 고객 데이터 분석을 통한 CLV 도출 및 해지 고객 예측
## 통신사 고객 데이터 분석을 통해 CLV를 도출하고 해지 고객을 예측해본다.
"""
데이터 설명
- 해지 여부
    Churn - 고객이 지난 1개월 동안 해지했는지 여부(Yes or No)
- Demographic 정보
    customerID - 고객들에게 배정된 유니크한 고객 번호
    gender - 고객의 성별 입니다(male or a female). 
    Age - 고객의 나이 입니다. 
    SeniorCitizen - 고객이 senior 시민인지 여부(1, 0). 
    Partner - 고객이 파트너가 있는지 여부(Yes, No).
    Dependents - 고객이 dependents가 있는지 여부(Yes, No). 

- 고객의 계정 정보
    tenure - 고객이 자사 서비스를 사용한 개월 수. 
    Contract - 고객의 계약 기간 (Month-to-month, One year, Two year)
    PaperlessBilling -  고객이 paperless billing를 사용하는지 여부 (Yes, No)
    PaymentMethod - 고객의 지불 방법 (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))

    MonthlyCharges - 고객에게 매월 청구되는 금액
    TotalCharges - 고객에게 총 청구된 금액 

- 고객이 가입한 서비스 
    PhoneService - 고객이 전화 서비스를 사용하는지 여부(Yes, No). 
    MultipleLines - 고객이 multiple line을 사용하는지 여부(Yes, No, No phone service). 
    InternetService - 고객의 인터넷 서비스 사업자 (DSL, Fiber optic, No). 
    OnlineSecurity - 고객이 online security 서비스를 사용하는지 여부 (Yes, No, No internet service)
    OnlineBackup - 고객이 online backup을 사용하는지 여부 (Yes, No, No internet service)
    DeviceProtection - 고객이 device protection에 가입했는지 여부 (Yes, No, No internet service)
    TechSupport 고객이 tech support를 받고있는지 여부 (Yes, No, No internet service)
    StreamingTV - 고객이 streaming TV 서비스를 사용하는지 여부 (Yes, No, No internet service)
    StreamingMovies - 고객이 streaming movies 서비스를 사용하는지 여부 (Yes, No, No internet service)

# 문제 정의
- 분석의 목적
    통신사의 고객 데이터에서 CLV를 계산한다.
    통신사 고객의 churn 해지를 예측한다.

"""

df = pd.read_csv('carrier_data.csv')
print(df.tail())
print(df.info())
print('빈 값 존재하는지 확인 : ', df.isnull().sum())

# 데이터를 확인해보니 TotalCharges 컬럼이 object 인 것이 이상하다. float으로 변환한다.
## pd.to_numeric(df['TotalCharges'])     pd.numeric으로 변환하려고 했으나 빈 칸이 존재하여 해당 함수를 사용할 수 없다.

## 빈 값을 확인해본다.
print(df[df['TotalCharges'] == " "])

# 위 빈 값을 확인해본 결과 tenure 값이 0인 값들이 빈칸으로 존재하고 있다.
# 즉 아직 가입한지 1개월이 지나지 않는 사람들이 한 번도 요금을 내지 않아 빈 칸인 것을 알 수 있다.
# 빈 값을 NaN으로 대체 해준다.
df['TotalCharges'] = df['TotalCharges'].replace(" ",np.nan)

"""
우리는 churn 여부에 관심이 있다.
TotalCharges가 빈 값인, tenure이 0인 유저들에게는 큰 관심이 없다.
그리고 이 유저들은 전체 7042 유저들 중 11명에 불과하다. 전체의 0.156%에 불과하다
TotalCharges의 null값을 버리고 실수로 변환한다.
"""

# null 값을 버린다.
df = df[df["TotalCharges"].notnull()]

df["TotalCharges"] = df["TotalCharges"].astype(float)

print(df.tail())

# 기술 통계와 상관관계를 확인해보자
print(df.describe())
print(df.corr())

corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

# 해지한 고객 수 확인
sns.countplot(y='Churn', data=df)
plt.show()
print(df.info())

# 변수간 pairplot 확인, churn 기준으로 보고싶기 때문에 hue값을 준다.
# sns.pairplot(df, markers="+", hue="Churn", palette="husl")            # 현재 Selected KDE bandwidth is 0. Cannot estiamte density. 오류메세지 뜨는 중. 검색했을 때 결측값이 존재해서 그렇다는데 과정을 둘러봐도 해설 코드와 동일함.
# plt.show()

"""
- Pariplot으로 눈으로 볼 수 있는 관계가 존재
    tenure가 낮은 경우 churn이 많다. 즉, 최근 고객들이 더 많이 해지한다.
    어느정도 이상의 tenure이 되면 충성고객이 되어 churn하지 않는 것 같다.
    MonthlyCharges가 높은 경우 churn이 많다.
    tenure과 MonthlyCharges가 아마도 주요한 변수인 것 같아 보인다.
    scatter plot을 봐도 어느 정도 견계선이 보인다.
- pairplot으로 불 수 있는 관계가 많지는 않다.
    numeric variable이 많지 않기 때문
    categorical variable을 처리해준다.
"""

# 다음 컬럼들에 대해 'No internet service'를 간단하게 'No'로 변환하여 통합해준다.
replace_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in replace_cols:
    df[i] = df[i].replace({'No internet service' : 'No'})

print(df.tail())

# https://www.kaggle.com/jsaguiar/exploratory-analysis-with-seaborn
# barplot을 %으로 표현한 함수
def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()

# SeniorCitizen
barplot_percentages('SeniorCitizen')
## SeniorCitizen은 전체 고객의 16% 정도에 불과하지만 churn 비율은 훨씬 높다.(42% vs 23%)

# Dependents
barplot_percentages('Dependents')
## Dependent가 없는 경오 churn을 더 많이 한다.

# Partner
barplot_percentages('Partner')
## Partner가 없는 경우 churn을 더 많이 한다.

# MultipleLines'
barplot_percentages('MultipleLines')
## phone service를 사용하지 않는 고객의 비율은 적다.
## MultipleLines를 사용중인 고객의 churn 비율이 아주 약간 높다.

# InternetService
barplot_percentages('InternetService')
## 인터넷서비스가 없는 경우의 churn 비율은 매우 낮다.
## Fiber opptic을 사용중인 고객이 DSL 사용중인 고객들보다 churn 비율이 높다.

# 6개의 부가 서비스관련 시각화
cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df1 = pd.melt(df[df['InternetService'] != 'No'][cols]).rename({'value' : 'Has service'}, axis=1)            # melt : 데이터 재구조화(replace와 비슷함.)
plt.figure(figsize=(10, 4.5))
ax = sns.countplot(data=df1, x='variable', hue='Has service')
ax.set(xlabel='Additional service', ylabel='Num of customers')
plt.show()
## 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport' 부가서비스 사용자는 churn 하는 경우가 적다.
## 스트리밍 서비스 이용 고객 중 churn이 많은 것으로 보인다('StreamingTV', 'StreamingMovies')

# Contract 유형에 따른 월청구 요금과 해지여부 시각화
ax = sns.boxplot(x='Contract', y='MonthlyCharges', hue='Churn', data=df)
plt.show()
## 장기계약이고 월청구요금이 높을수록 해지율이 높은 것 같다.
## 전반적으로 월청구요금이 높을 때 해지가능성이 높아보인다.

# PaymentMethod 유형에 따른 월청구요금과 해지여부를 시각화
ax = sns.boxplot(x='PaymentMethod', y='MonthlyCharges', hue='Churn', data=df)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
## Mailed check는 상대적으로 월청구요금이 낮다.
## Mailed check

# tenure에 따른 고객수 계산
print(df['tenure'].value_counts().sort_index())
a = df['tenure'].value_counts().sort_index()
print(a.shape)      # 해당 값은 몇개월 사용중인 고객이 몇명인지 보여주는 값이다.

# tenure에 따른 고객수 시각화
plt.figure(1, figsize=(16, 5))
plt.plot(np.arange(1, 73), a, 'p')
plt.plot(np.arange(1, 73), a, '-', alpha=0.8)
plt.xlabel('tenure'), plt.ylabel('Number of customer')
plt.show()
## 6개월 이후 retention이 상당히 낮아진다는 것을 알 수 있다.(retention : 잔존율)
## 반면 장기 충성고객들은 70개월 이상 유지되고 있다. 소중한 고객들이다.

# CLV(Customer Lifetime Value; LTV)를 계산한다.
"""
CLV는 고객생애 가치를 말한다.
고객이 확보된 이후 유지되는 기간동안의 가치
CAC와 LTV는 반드시 트래킹해야 할 주요 지표라고 할 수 있다.
    CAC보다 LTV가 최소 3배 이상 높은 것이 이상적

- LTV(Lifetime value)
    고객당 월 평균 이익(Avg monthly revenue per customer) x 평균 고객 유지개월 수(months customer lifetime)
    고객당 월 평균 이익(Avg monthly revenue per customer) / 월 평균 해지율(Monthly churn)
    (Average Value of a Slae) x (Number of Repeat Transactions) x (Average Retention Time in Months or Years for a Typical Customer)
    PLC(제품수명주기 : Product Life Cycle) x ARPU(고객평균매출 : Average Revenue Per User)
    (고객당 월 평균 이익(Avg Monthly Revenue per Customer) x 고객당 매출 총 이익(Gross Margin per Customer)) / 월 평균 해지율(Monthly Churn Rate)
- CAC(Customer Acquisition Cost)
    전체 세일즈 마케팅 비용(Total sales and marketing expense) / 신규확보 고객 수(New customers acquired)
- LTC : CAC Ratio
    LTV/CAC
        1:1 더 많이 팔수록 더 많이 잃게 된다(손해)
        3:1 이상적인 비율(도메인마다 다를 수 있다.)
        4:1 좋은 비즈니스 모델
        5:1 충분한 이익을 볼 수 있는데 마케팅에 투자를 덜 하고 있는것으로 보인다.
"""

# LTV(Lifetime Value)
## 고객당 월 평균 이익(Avg monthly revenue per customer) x 평균 고객 유지개월 수(months customer lifetime)
print(df['MonthlyCharges'].mean() * df['tenure'].mean())        # 2100.873646970263

# 2100 / 3   -> 700   이상적인 LTV/CAC 값이 3:1 이므로 3으로 나눠본다.
"""
LTV는 2100달러이다.
CAC는 700달러 정도인 것이 이상적이다.
통신사의 CAC는 기기 보조금, 멤버십 혜택 등이 있다.
"""


# Churn 해지할 고객을 예측해보자

# customerID는 분석에 사용되면 안되기 때문에 제거한다.
df2 = df.iloc[:, 1:]

# binary 형태의 카테고리 변수를 numeric variable로 변경해준다. -> 우리의 목표는 churn이다.
## replace를 통해 Yes는 1로, No는 0으로 변경시킨다. (머신러닝 코드를 돌리기 위해)
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No', value=0, inplace=True)

# 모든 categorical 변수를 더미 변수화 시킨다.
df_dummies = pd.get_dummies(df2)
print(df_dummies.tail())

# dummy 변수화한 데이터를 사용한다.
y = df_dummies['Churn'].values

X = df_dummies.drop(columns=['Churn'])

# 변수 값을 0과 1 사이 값으로 스케일링 해준다.
from sklearn.preprocessing import MinMaxScaler

features = X.columns.values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features
print(X.shape)
print(X.tail())

# Train Test data 생성
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# 고객이 해지하였는지 해지하지 않았는지를 알아보기 위함이므로 logistic regression 알고리즘을 사용한다.
## logistic regression은 시그모이드 함수를 활용해 0과 1을 찾아내는 대표적인 알고리즘
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
result = model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)

print(metrics.accuracy_score(y_test, prediction_test))      # 정확도 확인
## 0.8075829383886256    -> 해지 할 고객을 80% 확률로 예측하였다는 의미
