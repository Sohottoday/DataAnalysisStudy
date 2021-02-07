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
비디오 게임산업에서 1-day retention은 게임이 얼마나 재미있고 중독적인지 평가하는 주유 메트릭
1-day retention이 높을 경우 손쉽게 가입자 기반을 늘려갈 수 있다.
"""