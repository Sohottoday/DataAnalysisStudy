
list_arr = [i for i in range(10000) if i%2==0]

mylist = [1, 2, 3, 4, 5, 6, 7, 8]
even = [i * 2 for i in mylist if i%2 == 0]      # 이런식으로 반복문을 통한 출력값에 *2 와 같이 추가 연산을 해줄 수 있다.
print(even)



import collections

dict_test = {1:2, 2:3, 'abc':7}

print(collections.Counter(dict_test))