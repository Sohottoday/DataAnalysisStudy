import numpy as np

# 신경망의 내적(가중치만 적용)

X = np.array([[1, 2], [3, 4]])          # 입력신호
W = np.array([[1, 3, 5], [2, 4, 6]])        # 1, 3, 5는 x1의 가중치 / 2, 4, 6은 x2의 가중치

Y = np.dot(X, W)

