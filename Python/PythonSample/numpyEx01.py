import numpy as np

data = np.array([[10, 20], [30, 40]])

# 모든 요소의 합
result = np.sum(data)
print(result)
print('-' * 20)

result = np.sum(data, axis=1)
print(result)
print('-' * 20)

result = np.mean(data)
print(result)
print('-' * 20)

result = np.min(data)
print(result)
print('-' * 20)

result = np.max(data)
print(result)
print('-' * 20)

result = np.max(data, axis=0)
print(result)
print('-' * 20)



# 1차원 배열을 형상 변경, 행렬 연산, 전치

a = np.array([-1, 3, 2, -6])
b = np.array([3, 6, 1, 2])
print(a.ndim)
print(a.shape)

A = np.reshape(a, [2, 2])
print(A.ndim)
print(A.shape)

B = np.reshape(b, [2, 2])
print(A)
print(B)

print('-' * 20)

# matmul : Matrix multiply
# 머신러닝에서 이미지를 픽셀로 나눠
result3_1 = np.matmul(A, B)
result3_2 = np.matmul(B, A)
print(result3_1)
print('-'*30)
print(result3_2)
print('-'*30)

b = np.reshape(b, [1, 4])
a = np.reshape(a, [1, 4])
b2 = np.transpose(b)        # 전치

print(a)
print('-'*30)

print(b)
print('-'*30)

result3_3 = np.matmul(a, b2)
#result3_3 = np.matmul(a, b)
# matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1 is different from 4)
# 행렬의 모양이 같아야 한다.
print(result3_3)
print('-'*30)

print(b2)
print('-'*30)

