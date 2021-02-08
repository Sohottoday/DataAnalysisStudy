# A/B Test로 고객 retention을 높이자
## 모바일 게임의 고객 로그 데이터를 분서갷서 고객 유지율을 높이자.

"""
# 데이터 설명
- userid : 개별 유저들을 구분하는 식별 전호
- version : 유저들이 실험군 대조군 중 어디에 속했는지 알 수 있다.(gate_30, gate_40)
- sum_gamerounds : 첫 설치 후 14일 간 유저가 플레이한 라운드의 수
- retention_1 : 유저가 설치 후 1일 이내에 다시 돌아왔는지 여부
- retention_7 : 유저가 설치 후 7일 이내에 다시 돌아왔는지 여부

# 문제 정의
    Cookie Cats 게임에서는 특정 스테이지가 되면 스테이지가 Lock 되게 한다.
    Area Locked일 경우 Keys를 구하기 위한 특별판 게임을 해서 키 3개를 구하거나, 페이스북 친구에게 요청하거나, 유료아이템을 구매하여 바로 열 수 있다.
    Lock을 몇 번째 스테이지에 할 때 이용자 retention에 가장 좋을지 의사결정을 해야한다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cookie_cats.csv')
print(df.tail())
print("shape : ", df.shape)
print(df.info())

# AB 테스트로 사용된 버전별로 유저들을 몇 명씩 있을까?
print(df.groupby('version').count())            # 유저가 게임을 설치하면 gate_30 또는 gate_40 그룹으로 나뉘게 되었는데, 각 그룹별 유저는 거의 유사한 숫자로 배정

# 라운드 진행 횟수를 시각화 해보았다.
sns.boxenplot(data=df, y='sum_gamerounds')
plt.show()
"""
위 boxplot을 봤을 때 확실히 아웃라이어가 있는 것으로 보인다.
첫 14일동안 50,000회 가까이 게임을 한 사람들이 분명히 있지만 일반적인 사용행태라고는 하기 어렵다.
엄청나게 skewed한 데이터 분포
"""
# 아웃라이어 값이 하나이므로 제거해준다.
df = df[df['sum_gamerounds'] < 45000]

# percentile을 살펴보자(분위수)
print(df['sum_gamerounds'].describe())
## 상위 50%의 유저들은 첫 14일 동안 게임을 16회 플레이했다.


# 데이터 분석
## 각 게임실행횟수 별 유저의 수를 카운트 해본다.
print(df.groupby('sum_gamerounds')['userid'].count())
plot_df = df.groupby('sum_gamerounds')['userid'].count()

plot_df[:100].plot(figsize=(10,6))
plt.show()

"""
게임을 설치하고 한 번도 실행하지 않은 유저들의 수가 상당하다는 것을 알 수 있다.
몇몇 유저들은 설치 첫 주에 충분히 실행해보고 게임에 어느정도 중독되었다는 것을 알 수 있다.
비디오 게임산업에서 1-day retention은 게임이 얼마나 재미있고 중독적인지 평가하는 주요 메트릭
1-day retention이 높을 경우 손쉽게 가입자 기반을 늘려갈 수 있다.
"""

## 1-day retention의 평균을 살펴보자.
print(df["retention_1"].mean())     # 이러한 경우 True가 1 False가 0으로 계산된다. 즉, 절반정도는 게임을 설치한 날 게임을 했다는 의미

## 그룹별 1-day retention의 평균을 살펴보자.
print(df.groupby('version')['retention_1'].mean())
"""
단순히 그룹간 평균을 비교해봐서는 게이트가 40(44.2%)인 것보다 30(44.8%)인 경우에 플레이 횟수가 더 많다.
작은 차이지만 이 작은 차이가 retention, 더 나아가 장기적 수익에도 영향을 미치게 될 것이다.
그런데 이것만으로는 게이트를 30에 두는 것이 40에 두는 것보다 나은 방법이라고 확신할 수 없다.
"""

## 7-day retention의 평균을 살펴보자
print(df["retention_7"].mean())

## 그룹별 7-day retention의 평균을 살펴보자
print(df.groupby('version')['retention_7'].mean())
"""
단순히 그룹간 평균을 비교해봐서는 게이트가 40(18.2%)인 것보다 30(19.0%)인 경우에 생존률이 더 높다.
작은 차이지만 이 작은 차이가 retention, 더 나아가 장기적 수익에도 영향을 미치게 될 것이다.
1일보다 7일일때 차이가 더 크다. 그런데 이것만으로는 30에 두는 것이 40에 두는 것보다 나은 방법이라고 확신할 수 없다.
"""

# 부트스트랩을 활용하여 A/B test

# 각각의 AB그룹에 대해 bootstrap된 means 값의 리스트를 만듭니다.

boot_1d = []
for i in range(1000):
    boot_mean = df.sample(frac = 1,replace = True).groupby('version')['retention_1'].mean()
    boot_1d.append(boot_mean)
    
# list를 DataFrame으로 변환합니다. 
boot_1d = pd.DataFrame(boot_1d)
    
# 부트스트랩 분포에 대한 Kernel Density Estimate plot
boot_1d.plot(kind='density')
"""
잘 쓰게 될 방법은 아니다.(속도가 느림)
위의 두 분포는 AB 두 그룹에 대해 1 day retention이 가질 수 있는 부트 스트랩 불확실성을 표현한다.
비록 작지만 차이의 증거가 있는 것 같아 보인다.
자세히 살펴보기 위해 % 차이를 그려본다.
"""

# 두 AB 그룹간의 % 차이 평균 컬럼을 추가
boot_1d['diff'] = (boot_1d.gate_30 - boot_1d.gate_40)/boot_1d.gate_40*100

# bootstrap % 차이를 시각화
ax = boot_1d['diff'].plot(kind='density')
ax.set_title('% difference in 1-day retention between the two AB-groups')
plt.show()

# 게이트가 레벨30에 있을 때 1-day retention이 클 확률을 계산
print('게이트가 레벨30에 있을 때 1-day retention이 클 확률:',(boot_1d['diff'] > 0).mean())
"""
위 도표에서 가장 가능성이 높은 % 차이는 약 1 ~ 2% 이며 분포의 95%는 0% 이상이며 레벨 30의 게이트를 선호한다.
부트스트랩 분석에 따르면 게이트가 30에 있을 때 1일 유지율이 더 높을 가능성이 높다.
그러나 플에이어는 하루 동안만 게임을 했기 때문에 대부분의 플레이어가 아직 레벨 30에 다다르지 않았을 가능성이 크다.
즉, 대부분의 유저들은 게이트가 30에 있는지 여부에 따라 retention이 영향받지 않았을 것이다.
일주일동안 플레이 한 후에는 더 많은 플레이어가 레벨 30과 40에 도달하기 때문에 7일 retention도 확인해야 한다.
"""

df.groupby('version')['retention_7'].sum() / df.groupby('version')['retention_7'].count()
"""
1일 retention과 마찬가지로, 게이트가 30 레벨(19.0%)에 있는 경우보다 게이트 레벨이 40(18.2%)인 경우 7일 retention이 낮다.
이 차이는 1일 retention보다 차이가 더 큰데, 아마도 더 많은 플레이어가 첫 번째 게이트를 열어볼 시간이 있었기 때문
전체 7일 retention은 전체 1일 retention보다 낮다. 설치 후 하루보다 설치 후 일주일에 게임을 하는 사람이 더 적기 때문
이전과 마찬가지로 부트 스트랩 분석을 사용하여 AB 그룹간의 차이가 있는지 확인한다.
"""

# 각각의 AB그룹에 대해 bootstrapp된 means 값의 리스트를 만듭니다.
boot_7d = []
for i in range(500):
    boot_mean = df.sample(frac=1,replace=True).groupby('version')['retention_7'].mean()
    boot_7d.append(boot_mean)
    
# list를 DataFrame으로 변환합니다. 
boot_7d = pd.DataFrame(boot_7d)

# 두 AB 그룹간의 % 차이 평균 컬럼을 추가합니다.
boot_7d['diff'] = (boot_7d.gate_30 - boot_7d.gate_40)/boot_7d.gate_40*100

# bootstrap % 차이를 시각화 합니다.
ax = boot_7d['diff'].plot(kind='density')
ax.set_title('% difference in 7-day retention between the two AB-groups')
plt.show()

# 게이트가 레벨30에 있을 때 7-day retention이 더 클 확률을 계산합니다. 
print('게이트가 레벨30에 있을 때 7-day retention이 클 확률:',(boot_7d['diff'] > 0).mean())

"""
    부트 스트랩 결과는 게이트가 레벨 40에 있을 때보다 레벨 30에 있을 때 7일 retention이 더 높다는 강력한 증거가 있음을 나타낸다.
    결론은 retention을 늘리기 위해서 게이트를 레벨 30에서 40으로 이동해서는 안된다.
"""