# while : 조건식이 참(True)인 동안 반복
# while <조건식>:
#   문1
# else:
#   문2
value = 5
while value > 0:
    print(value)
    value -= 1

# for in : 이터레이션이 가능한 객체를 순차적으로 순회
# for <타겟> in <객체>:
#   문1
# else:
#   문2
I = ['Apple', 100, 15.23]
for i in I:
    print(i,type(i))
d = {"Apple":100, "Orange":200, "Banana":300}
for k, v in d.items():      # 이와 같은 형식으로 딕셔너리도 표현 가능.
    print(k,v)

# 제어문 : break, continue 그리고 else: 반복문을 수행하면서 break문과 continue문을 이용해서
# 반복구간을 탈출하거나 아래쪽 라인을 스킵할 수 있음.
L = [1,2,3,4,5,6,7,8,9,10]
for i in L:
    if i > 5 :
        break
    print("Item:{0}".format(i))     #{0} = {0}자리에 i값이 들어가라는 뜻

for i in L:
    if i % 2 ==0 :
        continue
    print("Item:{0}".format(i))

