import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.csv'
data = pd.read_csv(filename)

print(data['class'].unique())

inputData = ['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age']
x_data = data[inputData]
y_data = data['class']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# 모델 생성하기
model = LogisticRegression()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
print(f'# train 정확도 : {train_score}')

test_score = model.score(x_test, y_test)
print(f'# test 정확도 : {test_score}')

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ')
print(type(model.coef_))
coef = model.coef_.ravel()

maxidx = np.argmax(coef.tolist())
print('가중치가 가장 큰 컬럼 : ', inputData[maxidx])

print(coef.tolist())

mylist = []     # 컬럼 이름과 가중치 정보를 담고 있는 리스트
for idx in range(len(coef.tolist())):
    mylist.append((inputData[idx], coef.tolist()[idx]))
print(mylist)

print('절편 : ')
print(model.intercept_)

print('test result:')

prediction = model.predict(x_test)
print('confusion matrix : ')

cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
print('-' * 40)

accuracy = accuracy_score(y_test, prediction)
print(f'\n정확도 : {100 + accuracy}')
print('\nclassification report : ')
cl_report = classification_report(y_test, prediction)
print(cl_report)
print('-' * 40)

# 히트맵 생성

plt.rc('font', family='Malgun Gothic')

sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap='YlGnBu', fmt='g')

plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predict label')

plt.savefig('logisticRegression03_01.png', dpi=400, bbox_inches='tight')



