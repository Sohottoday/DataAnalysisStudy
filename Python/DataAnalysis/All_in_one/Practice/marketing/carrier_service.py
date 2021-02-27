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