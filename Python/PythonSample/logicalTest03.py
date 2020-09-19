
# 훈련 데이터와 테스트 데이터를 7대3으로 분리한다.
# 최종 결과에 대하여 정확도를 구해본다.

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf

filename = 'logicalTest03.csv'
data = np.loadtxt(filename, delimiter=',', dtype=np.int32)

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column:]

seed = 1234
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='sigmoid'))

learning_rate = 0.01
sgd = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, epochs=200, verbose=1)

total, hit = 0, 0       # 총 개수, 맞춘 개수

for idx in range(len(x_test)):
    result = model.predict_classes(np.array([x_test[idx]]))
    print(f'테스트용 데이터 : {x_test[idx]}')
    print(f'정답 : {y_test[idx]}', end=' ')
    print(f'예측 값 : {str(result.flatten())}')

    total += 1

    # 예측 값과 정답이 같은 경우 1 추가
    hit += int(y_test[idx] == result.flatten())
    print('-' * 30)

# end for
accuracy = hit/total
print(f'정확도 = {accuracy:.4f}')

print('finished')

