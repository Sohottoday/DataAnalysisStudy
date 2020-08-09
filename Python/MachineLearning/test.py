import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 신경망의 내적(가중치만 적용)

X = np.array([[1, 2], [3, 4]])          # 입력신호
W = np.array([[1, 3, 5], [2, 4, 6]])        # 1, 3, 5는 x1의 가중치 / 2, 4, 6은 x2의 가중치

Y = np.dot(X, W)


# 신경망 파이썬 구현(sigmoid 함수 사용)

print(Y)

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
print(A1)

z1 = sigmoid(A1)
print(z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(z1, W2) + B2
print(A2)

Z2 = sigmoid(A2)
print(Z2)


def identity_function(x):           # 항등 함수(시그마 함수)
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3)           # Y = A3
print(Y)
print(A3)


# 다층 신경망

## 가중치와 편향을 초기화해주는 함수

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

## 입력 신호를 출력으로 변환하는 처리과정(순방향)

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])

y = forward(network, x)
print(y)






