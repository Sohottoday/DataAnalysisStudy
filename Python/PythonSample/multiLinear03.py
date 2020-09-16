import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = np.loadtxt('multiLinear03.csv', delimiter=',')

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=5000, batch_size=10, verbose=0)

print(model.get_weights())
print('-' * 30)

prediction = model.predict(x_test)

for idx in range(len(y_test)):
    real = y_test[idx]
    pred = prediction[idx]
    print(f'real : {real}, pred : {pred}')

print('finished')

