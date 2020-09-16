# 사이킷-런을 이용한 회귀 분석
# 결정 계수 : 실제 값이 예측 값과 얼마 정도의 일치성을 보이는지를 나타내는 척도
# 값은 0부터 1 사이의 값으로, 1에 가까우면 설명력이 좋다 라고 표현한다.
# R-squared = 1 - Σ(y-H) ** 2 / Σ(y-bary) ** 2
# y는 실제 정답, H는 가설로 도출된 값, bary는 실제 정답의 평균 값

import numpy as np
from sklearn.linear_model import LinearRegression       # 선형 회귀
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

# 학습용 데이터 셋 : 'linearTest01.csv'
# 테스트용 데이터 셋 : 'linearTest02.csv
filename = 'linearTest01.csv'

# skiprows : 머리글 1행은 제외
training = np.loadtxt(filename, delimiter=',', skiprows=1)
print(training)

x_column = training.shape[1] - 1        # 선형에서 했던 table_col = training.shape[1], y_column = 1 이러한 것들을 한번에 표현한 것

x_train = training[:, 0:x_column]
y_train = training[:, x_column:]

# 모델 객체 생성
model = LinearRegression()

model.fit(x_train, y_train)     # 학습

print(f'기울기(w) : {model.coef_}')        # w 값
print(f'절편(b) : {model.intercept_}')        # b 값

# residual(잔차)
print(f'잔차의 제곱합(cost) : {model._residues}')

# 시각화
plt.title('그래프')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train, y_train, 'bo')

train_pred = model.predict(x_train)
plt.plot(x_train, train_pred, 'r')      # 회귀 선


pngname = 'linearRegression.png'
plt.savefig(pngname)

print(f'{pngname}이 저장되었습니다.')

filename = 'linearTest02.csv'

testing = np.loadtxt(filename, delimiter=',', skiprows=1)
print(testing)

x_column = testing.shape[1] - 1        # 선형에서 했던 table_col = training.shape[1], y_column = 1 이러한 것들을 한번에 표현한 것

x_test = testing[:, 0:x_column]
y_test = testing[:, x_column:]

# 산술 연산에 의한 결정 계수 구하기
y_test_mean = np.mean(np.ravel(y_test))

# TSS : 편차의 제곱의 총 합(Total sum of square)
TSS = np.sum((np.ravel(y_test)-y_test_mean)**2)

# RSS : 회귀식과 평균값의 차이의 총 합(Residual sum of squares)
RSS = np.sum((np.ravel(y_test)-np.ravel(model.predict(x_test)))**2)

# 결정 계수 = 1 - RSS / TSS
R_Squared = 1 - (RSS/TSS)

print(f'R_Squared 01 : {R_Squared}')

print(f'R_Squared 02 : {model.score(x_test, y_test)}')


