import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno
import folium
import numpy as np
import glob

import matplotlib.font_manager as fm

path = 'C:\\Users\\user\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumSquareRoundEB.ttf'
font_name = fm.FontProperties(fname=path, size=14).get_name()
print(font_name)
plt.rc('font', family=font_name)

# 설정한 폰트는 rebuild 해줘야 적용된다.
fm._rebuild()

df_store = pd.read_csv('C:\\Users\\user\\Desktop\\Programing\\Github\\SelfStudy\\Python\\DataAnalysis\\All_in_one\\Practice\\marketing\\소상공인시장진흥공단_상가(상권)정보_제주_202012.csv', encoding='utf-8', delimiter='|')

print(df_store.shape)
print(df_store.columns)

