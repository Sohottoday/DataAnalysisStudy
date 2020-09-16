import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

filename = 'logicalTest02.csv'
data = np.loadtxt(filename, delimiter=',')

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x_train = data[:, 0:x_column]
y_train = data[:, x_column:]

model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='sigmoid'))

learning_rate = 0.1
sgd = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, epochs=2000, verbose=1)

# 0 : 강아지, 1 : 고양이
x_test = [[2, 1], [6, 5], [11, 6]]

def getCategory(mydata):
    mylist = ['강아지', '고양이']
    print(f'예측 : {mydata}, {mylist[mydata[0]]}')

H = model.predict(x_train)
print(H)
print('-'*30)

for idx in x_test:
    HH = model.predict(np.array([idx]))
    # flatten() : 차원을 1차원으로 만들어 주는 함수
    print(HH.flatten())
    print(HH)
    print('#'*30)

    pred = model.predict_classes(np.array([idx]))
    print('테스트 데이터 : ', np.array([idx]))
    getCategory(pred.flatten())
    print('*'*30)

# 이러한 식으로 분류에 관한 문제를 해결 가능하다.



