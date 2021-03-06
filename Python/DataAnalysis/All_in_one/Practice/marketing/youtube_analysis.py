# youtube Trend


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

KRvideo = pd.read_csv('KRvideos.csv', engine='python', error_bad_lines='false')       # error_bad_lines : 에러가 나는 라인은 무시하고 불러오겠다는 의미
print(KRvideo.head())
# codec can't decode byte 0xec in position 232: illegal multibyte sequence         라고 오류가 뜨는데 왜 뜨는지 이유를 알 수가 없음.

print(KRvideo.shape)

# 결측 값 확인
print(KRvideo.isnull().sum())


df = KRvideo[['title','channel_title','views']]
df_sorted = df.sort_values(by='views', ascending=False)
print(df_sorted)

