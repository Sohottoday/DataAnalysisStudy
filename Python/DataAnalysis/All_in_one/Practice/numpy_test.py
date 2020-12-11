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
