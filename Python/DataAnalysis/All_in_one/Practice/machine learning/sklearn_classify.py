import warnings

# 불필요한 경고 출력을 방지한다.
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 데이터 로드
from sklearn.datasets import load_iris

iris = load_iris()          # 데이터셋은 딕셔너리 형태로 되어있다.
"""
- DESCR : 데이터셋의 정보를 보여준다.
- feature_names : feature data의 컬럼 이름
- target : label data(수치형)
- target_names : label의 이름(문자형)
"""

print(iris['DESCR'])
data = iris['data']
feature_names = iris['feature_names']
print(feature_names)        # sepal : 꽃 받침, petal : 꽃잎
target = iris['target']

# 데이터프레임 만들기
df_iris = pd.DataFrame(data, columns=feature_names)
print(df_iris.head())

df_iris['target'] = target

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot('sepal width (cm)', 'sepal length (cm)', hue='target', palette='muted', data=df_iris)
plt.title('Sepal')
plt.show()

# 훈련
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'])        # split 뒤 첫번째는 feature data, 두번째는 target data
print("훈련 : ",x_train.shape, y_train.shape)
print("valid : ", x_valid.shape, y_valid.shape)
## 데이터를 분배하여 학습시킬 때 종종 클래스들의 분포가 불균형적인 경우가 있는데
## 이러한 경우에는 stratify : label로 균등하게 분배해준다.
x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])
