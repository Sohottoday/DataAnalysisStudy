# 배열 생성하는 여러가지 방법

import numpy as np

# zeros : 요소가 0인 배열/행렬을 생성해주는 함수
print(np.zeros(3))

arr = np.zeros((2, 2))
print(arr)

# ones : 요소가 1인 배열/행렬을 생성
arr2 = np.ones((3, 2))
print(arr2)

# 연립 방정식을 풀고자 할 때 사용(가우스 소거법)
# 크기가 3인 단위 행렬을 만든다.
# 정방 행렬 : 행과 열의 크기가 같은 행렬
arr3 = np.eye(3)
print(arr3)

arr4 = np.full((2, 2), 5)
print(arr4)