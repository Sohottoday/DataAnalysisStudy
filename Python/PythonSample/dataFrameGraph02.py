import pandas as pd
import matplotlib.pyplot as plt

filename = 'ex802.csv'

plt.rc('font', family = 'Malgun Gothic')

myframe = pd.read_csv(filename, encoding='utf-8', index_col='type')
myframe.index.name = '자동차 유형'
myframe.columns.name = '도시(city)'
print(myframe)

myframe.plot(kind='bar', rot=0, title='차량 유형별 지역 등록 댓수', legend=True)

#plt.legend(loc='best')

filename = 'graph01.png'
plt.savefig(filename)

myframeT = myframe.T
print(myframeT)

myframeT.plot(kind='bar', rot=0, title= '지역별 차량 유형 등록 댓수')
filename = 'graph02.png'
plt.savefig(filename)

myframeT.plot(kind='bar', rot=0, title= '지역별 차량 유형 등록 댓수', stacked=True)
filename = 'graph03.png'
plt.savefig(filename)