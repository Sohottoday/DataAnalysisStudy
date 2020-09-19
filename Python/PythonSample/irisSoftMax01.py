import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

filename = 'newiris.csv'

df = pd.read_csv(filename)
print(df.columns)
print('-' * 30)

print(df['Species'].unique())
print('-' * 30)

data = df.values

table_col = data.shape[1]
print(table_col)
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]

# 주의 : 정답은 현재 문자열로 되어 있습니다.
y_imsi = data[:, x_column:]
#print(y_imsi)
#print('-' * 30)

# 1차원 형식으로 만들기
y_imsi = y_imsi.ravel()
#print(y_imsi)

# y_imsi 컬럼은 문자열인데, 이것을 숫자형 데이터로 변환한다.
le = LabelEncoder()
le.fit(y_imsi)
y = le.transform(y_imsi)
print(y)

x = x.astype(np.float)
y = y.astype(np.float)

# 데이터 셋 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 원핫 인코딩
NB_CLASSES = 3      # 예측할수 있는 품종이 3개 이므로

y_train = np_utils.to_categorical(y_train, NB_CLASSES, dtype='float32')

print(y_train)

# 모델 생성
model = Sequential()

model.add(Dense(units=NB_CLASSES, input_shape=(x_column, ), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000, verbose=0)
print(history.history.keys())
print('-' * 30)

print(history.history['loss'])
print('-' * 30)

print(history.history['accuracy'])
print('-' * 30)

# 예측값과 정답 데이터를 저장할 리스트
total_list = []

for idx in range(len(x_test)):
    H = model.predict(np.array([x_test[idx]]))
    pred = np.argmax(H, axis=-1)
    print('예측 값 : ', pred, end=' ')
    print(f'정답 : {int(y_test[idx])}')
    print('가설 정보 : ', H.flatten())
    print('-' * 30)

    # item() 함수는 numpy 배열을 파이썬 형태로 변경해주는 함수
    total_list.append((float(pred.item()), y_test[idx]))

myframe = pd.DataFrame(total_list, columns=['예측 값', '정답'])
filename = 'irisSoftMaxResult.csv'
myframe.to_csv(filename, index=False)
print(filename + '이 저장되었습니다.')
