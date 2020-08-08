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