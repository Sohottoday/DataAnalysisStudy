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