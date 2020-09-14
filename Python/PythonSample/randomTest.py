import numpy as np

print('임의의 값으로 채워진 행렬 생성')
# random : 최소 0 이상 1 미만의 숫자
result = np.random.random((2, 2))
print(result)
print('-' * 20)

print('표준 편차가 1, 평균이 0인 정규 분포에서 표본 추출')
# randn : Random Normalization
result = np.random.randn(2, 3)
print(result)
print('-' * 20)

print('임의의 값으로 채워진 배열 생성')
result = np.random.rand(4, 4)
print(result)
print('-' * 20)

print('균등 분포에서 데이터 추출')
result = np.random.uniform(size = (3, 3))
print(result)
print('-' * 20)

print('정수 0이상 x미만의 정수 추출')
result = np.random.randint(5)
print(result)
print('-' * 20)

result = np.random.randint(3, size=4)
print(result)
print('-' * 20)

result = np.random.randint(0, 5, size=10)
print(result)
print('-' * 20)

# 0, 1, 2 중에서 임의로 하나를 추출하는 동작을 5번 수행하여 나온 수의 총 합을 구해보시오.

randnum = np.random.randint(0, 3, size=5)
print(randnum)
result = np.sum(randnum)
print(result)

print('-'*20)

print('permutation은 0부터 length까지의 수를 임의로 섞어 준다.')
length = 10
result = np.random.permutation(length)
print(result)
print('-'*20)

# 시드 배정은 동일한 데이터를 샘플링하거나 머신러닝시 동일한 결과를 한시적으로 추출해보고자 할 때 사용한다.
seed = 100
np.random.seed(seed)    # 랜덤 값에 시드 배정

# 0 <= 값 < 5 사이의 값 3개를 추출
#result = np.random.choice(5, 3)
#print(result)
print('-'*20)

result = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
print(result)
print('-'*20)