import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/Pokemon.csv")
print(df.head())
"""
Name : 포켓몬 이름
Type 1 : 포켓몬 타입 1
Type 2 : 포켓몬 타입 2
Total : 포켓몬 총 능력치(Sum of attack, Sp.Atk, Defense, Sp.Def, Speed and HP)
HP : 포켓몬 HP 능력치
Attack : 포켓몬 Attack 능력치
Defense : Defense 능력치
Sp.Atk : Sp.Atk 능력치
Sp.Def : Sp.Def 능력치
Speed : Speed 능력치
Generation : 포켓몬 세대
Legendary : 전설 포켓몬 여부
"""

# EDA (Exploratory Data Analysis : 탐색적 데이터 분석)
print(df.shape)
print(df.info())
print(df.isnull().sum())

# 개별 피처 탐색
print(df['Legendary'].value_counts())

df['Generation'].value_counts().sort_index().plot()
plt.show()

print("Type 1 : ", df['Type 1'].unique())
print("Type 2 : ", df['Type 2'].unique())
print(len(df[df['Type 2'].notnull()]['Type 2'].unique()))

# 데이터 특징 탐색
## 변수들의 분포 탐색
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
sns.boxplot(data=df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']], ax=ax)
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
sns.boxplot(data=df[df['Legendary']==1][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']], ax=ax)
plt.show()
## 위의 방법을 통해 레전더리 포켓몬들은 기본 능력치가 평균보다 높다는 것을 알 수 있다.

df['Total'].hist(bins=50)
plt.show()

## Legendary 그룹별 탐색
df['Type 1'].value_counts(sort=False).sort_index().plot.barh()
plt.show()

df[df['Legendary']==1]['Type 1'].value_counts(sort=False).sort_index().plot.barh()
plt.show()

df['Type 2'].value_counts(sort=False).sort_index().plot.barh()
plt.show()

df[df['Legendary']==1]['Type 2'].value_counts(sort=False).sort_index().plot.barh()
plt.show()

df['Generation'].value_counts(sort=False).sort_index().plot.barh()
plt.show()

df[df['Legendary']==1]['Generation'].value_counts(sort=False).sort_index().plot.barh()
plt.show()      # 각 세대별 레전더리 포켓몬 수

groups = df[df['Legendary']==1].groupby('Generation').size()
groups.plot.bar()
plt.show()

## 포켓몬 능력 분포 탐색
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
sns.boxplot(x = "Generation", y='Total', data=df, ax=ax)
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
sns.boxplot(x = "Type 1", y='Total', data=df, ax=ax)
plt.show()


# 지도학습 기반 분류 분석
## 데이터 전처리
df['Legendary'] = df['Legendary'].astype(int)           # 타입 변경
df['Generation'] = df['Generation'].astype(int)         # 타입 변경
preprocessed_df = df[['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']]

## one-hot encoding
encoded_df = pd.get_dummies(preprocessed_df['Type 1'])

def make_list(x1, x2):
    type_list = []
    type_list.append(x1)
    if x2 is not np.nan:
        type_list.append(x2)
    return type_list

preprocessed_df['Type'] = preprocessed_df.apply(lambda x : make_list(x['Type 1'], x['Type 2']), axis=1)
print(preprocessed_df.head())

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
preprocessed_df = preprocessed_df.join(pd.DataFrame(mlb.fit_transform(preprocessed_df.pop('Type')), columns=mlb.classes_))
# join을 통해 인코딩이 된 데이터들은 원래의 데이터프레임에 붙여준다.
# MultiLabelBinarizer을 활용하면 원핫 인코딩을 멀티로 적용시켜 준다.

preprocessed_df = pd.get_dummies(preprocessed_df['Generation'])

## 피처 표준화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scale_columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
preprocessed_df[scale_columns] = scaler.fit_transform(preprocessed_df[scale_columns])
# 위의 스케일링은 Z스케일링으로 음의값도 나온다. 만약 0~1의 값으로만 표준화시키는 MinMax 스케일러를 활용한다면 더 좋은 결과를 나타낼 수 있다고 생각할 수 있다.

## 데이터셋 분리
from sklearn.model_selection import train_test_split

x = preprocessed_df.loc[:, preprocessed_df.columns != 'Legendary']
y = preprocessed_df['Legandary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

