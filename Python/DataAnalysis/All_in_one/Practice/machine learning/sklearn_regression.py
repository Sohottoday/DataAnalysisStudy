# 회귀(regression) 예측
"""
수치형 값을 예측(Y의 값이 연속된 수치로 표현)
ex)
    주택 가격 예측
    매출액 예측
"""

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)          # 숫자를 표시할 때 e-5 이런식으로 표기하는것을 0.00001 과 같이 바꿔줌

from sklearn.datasets import load_boston

data = load_boston()

df = pd.DataFrame(data['data'], columns=data['feature_names'])

# Y 데이터인 price도 데이터 프레임에 추가
df['MEDV'] = data['target']

print(df.head())