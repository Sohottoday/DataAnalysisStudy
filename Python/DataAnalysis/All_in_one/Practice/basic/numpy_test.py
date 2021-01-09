import numpy as np

arr = np.array([1, 2, 3, 4], dtype=int)
print(arr)

aaa = [1, 2, 3, 4]
print(type(arr))
print(type(aaa))

mylist1 = [1, 2, 3, 4]
mylist2 = [[1, 2, 3, 4],
           [5, 6, 7, 8]]

arr1 = np.array(mylist1)
arr2 = np.array(mylist2)

print(arr1.shape)
print(arr2.shape)

# array는 list와 다르게 1개의 단일 데이터 타입만 허용된다.

# int와 float 타입이 혼재된 경우
arr = np.array([1, 2, 3, 3.14])
print(arr)

# int와 float 타입이 혼재되었으나 dtype을 지정한 경우
arr = np.array([1, 2, 3, 3.14], dtype=int)
print(arr)

# int와 str 타입이 혼재된 경우
arr = np.array([1, 3.14, '굿굿', '1234'])
print(arr)          # 모두 문자열로 인식한다.
# str과 혼재된 경우 강제로 dtype을 int로 변경하면 에러가 발생한다.
# 단 문자열이 숫자로만 구성된 문자열이라면 python에서 알아서 숫자로 변환시켜 인식한다.
arr = np.array([1, 3.14, '1234'], dtype=int)
print(arr)


# 2차원 array
arr2d = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])
print(arr2d.shape)
print(arr2d[0, 2])      # 0행 2열
print(arr2d[2, 1])

print(arr2d[0, : ])     # 행(row)을 모두 가져오려는 경우
print(arr2d[ : , 2])    # 2번째 열(colunm)을 모두 가져오려는 경우
print(arr2d[:2, :])
print(arr2d[:2, 2:])

# fancy 인덱싱 : 범위가 아닌 특정 index의 집합의 값을 선택하여 추출하고 싶을 때 사용
arr = np.array([10, 23, 2, 7, 90, 65, 32, 66, 70])
print(arr[[1, 3, 5]])       # 이러한 방식으로 [추출하고 싶은 인덱스] 를 활용한다
idx = [1, 3, 5]
print(arr[idx])             # 이렇게 변수에 담아서도 활용이 가능하다

print(arr2d[[0, 1], :])     # 2차원에서는 앞은 행, 뒤는 열
print(arr2d[:, [1, 2, 3]])

# 조건을 활용한 인덱싱(Boolean 인덱싱)
print(arr2d>2)
# 위를 활용하여 []로 한번 더 묶어주면 된다.
print(arr2d[arr2d>2])
print(arr2d[arr2d<5])

# arange
arr = np.arange(1, 11)
print(arr)

arr = np.arange(1, 11, 2)
print(arr)

# 정렬 sort
arr = np.array([1, 10, 5, 8, 2, 4, 3, 6, 8, 7, 9])
print(np.sort(arr))     # 기본적으로 오름차순
print(np.sort(arr)[::-1])       # 내림차순은 이러한 방식으로 수행한다.
# 위의 np.sort는 변수 자체에 담지 않으면 유지가 되지 않는다. 따라서
arr.sort()     # python 내장 함수로 정렬할 경우 곧바로 정렬된다.
print(arr)
# N차원 정렬
arr2d = np.array([[5, 6, 7, 8],
                  [4, 3, 2, 1],
                  [10, 9, 12, 11]])
# 열 정렬(왼쪽에서 오른쪽으로)
print(np.sort(arr2d, axis=1))
# 행 정렬(위에서 아래로)
print(np.sort(arr2d, axis=0))
# index를 반환하는 argsort
print(np.argsort(arr2d, axis=1))

# matrix
# matrix의 곱셈은 맞닿는 shape가 같아야 한다.
# ex) 2X3과 2X3행은 맞닿는 부분이 3과 2이므로 불가능
# ex) 2X3과 3X2행은 맞닿는 부분이 3으로 같으므로 곱셈 가능 => 2X2 행의 결과값 출력 : 맞닿는 부분이 아닌 바깥쪽 크기 만큼의 결과값 출력됨
a = np.array([[1, 2, 3],
              [2, 3, 4]])

b = np.array([[3, 4, 5],
              [1, 2, 3]])

print(a+b)
print(a-b)
# 모양(shape)가 맞지 않을 경우 오류 발생
# operands could not be broadcast together with shapes (2, 3) (3, 2)
# matrix 내 행이나 열의 합계를 구할 때 sum 사용
print(np.sum(a, axis=0))        # 열끼리 더함
print(np.sum(a, axis=1))        # 행끼리 더함

# 일반 곱셈
print(a * b)
# 단순 곱셈과 dot product는 다르다.
a = np.array([[1, 2, 3],
              [2, 3, 4]])

b = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(np.dot(a, b))
print(a.dot(b))         # python 내장 함수인 dot 함수로도 행렬 dot product가 가능하다.

# broadcasting
# 행렬 자체에 전체적인 계산을 진행하고 싶을 때 사용
print(a + 3)
print(a * 3)
print(a / 3)
