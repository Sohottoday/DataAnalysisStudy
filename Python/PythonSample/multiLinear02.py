
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

filename = 'multiLinear02.csv'
data = np.loadtxt(filename, delimiter=',')

table_col = data.shape[1]       # 4
y_column = 1            # 출력(정답) 데이터 컬럼 수
x_column = table_col - 1        # 4 - 1 = 3

# 입력과 출력 데이터 분리
x = data[:, 0:x_column]
y = data[:, x_column:]

# 훈련용 데이터 셋과 테스트용 데이터 셋 분리
seed = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=seed)

# 모델 생성
model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5000, batch_size=10, verbose=0)
print(model.get_weights())

prediction = model.predict(x_test)

for idx in range(len(y_test)):
    label = y_test[idx]     # 실제 정답
    pred = prediction[idx]      # 예측 정답

    print(f'real : {label}, prediction : {pred}')

print('finished')

