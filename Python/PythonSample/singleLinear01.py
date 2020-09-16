import numpy as np

# sklearn : 과학kit(머신러닝을 하기 위한 패키지)
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


'''
첨부한 엑셀 파일을 읽어온다.
훈련/학습 데이터와 테스트 데이터의 비율을 75대 25로 분리한다.
훈련/학습 데이터를 이용하여 선형 회귀 분석을 한 다음, 테스트 데이터를 이용하여 결과를 예측한다.
'''

filename = 'singleLinear01.csv'
data = np.loadtxt(filename, delimiter=',')
print(type(data))

table_col = data.shape[1]       # 컬럼 수
#print(table_col)
y_column = 1        # 출력(정답) 데이터 컬럼 수
x_column = table_col - y_column     # 입력 데이터 컬럼 수

x = data[:, 0:x_column]
y = data[:, x_column:]
print(x)
print('-' * 30)

print(y)
print('-' * 30)

# 입력용 학습, 입력용 테스트, 출력용 학습, 출력용 테스트
#       = train_test_split(입력원본, 출력원본, test_size=테스트데이터의비율)
# 일반적으로 7:3 으로 나눈다.(절대적인 것은 아니다.)
x_train, x_test, y_train, y_test  \
    = train_test_split(x, y, test_size=0.25)

print(x_train)
print('-'*30)
print(y_train)
print('-'*30)

# 모델을 생성하고, 학습을 수행한다.
# 단계 01 : 모델(작업 공간) 생성
model = Sequential()

# 단계 02 : 레이어를 추가한다.
# Dense : 레이어를 추가하는데 사용되는 클래스
# units : 출력값의 크기, input_dim : 입력 데이터의 크기, activation : 활성화함수
# linear : 선형 회귀 분석
# 총 레이어 갯수  : add() 함수 갯수 + 1
model.add(Dense(units=1, input_dim=1, activation='linear'))

# 단계 03 : 컴파일을 수행한다.
# loss : 비용(손실) 함수를 지정합니다.
# optimizer : 옵티마이저로써, 경사 하강법을 잘 수행하기 위한 가이드
model.compile(loss='mean_squared_err', optimizer='adam')

# 단계 04 : 훈련/학습을 합니다. 기출 문제 풀기
# epochs : 반복 횟수
# batch_size : 1번에 수행할 데이터의 양
# verbose : 0(진행 결과를 출력하지 않음), 2(epoch당 1번), 1(기본 값)
# 교사 학습법 : 입력 데이터와 출력(정답) 데이터를 같이 넣어주는 학습 법
model.fit(x_train, y_train, epochs=5000, batch_size=10, verbose=1)

# 단계 05 : 예측을 합니다. 새로운 문제 풀기
prediction = model.predict(x_test)

for idx in range(len(y_test)):
    label = y_test[idx]     # 정답 데이터
    pred = prediction       # 예측치

    print('real : %f, prediction : %f' % (label, pred))

print('finished')
