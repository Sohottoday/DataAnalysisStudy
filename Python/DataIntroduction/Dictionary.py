# 딕셔너리(Dictionary) : 강력하면서 알아두면 편리한 자료구조
# 임의의 객체 집합적 자료형이며 별도의 자료 순서를 가지지 않고 자료의 순서를 정할 수 없는
#  맵핑(Mapping) 형이며 키를 통한 빠른 검색이 필요할 때 사용함.
# key와 value형이라고 생각하면 됨.

d = dict(a=1, b=2, c=3)
print(d)
print(type(d))
color = {"apple" : "red", "banana" : "yellow"}
print(color)

color["cherry"] = "red"
print("체리 변수 추가 : ",color)

for c in color.items():{
    print(c)
}

for k,v in color.items(): {
    print(k,v)
}

# 딕셔너리에서 데이터를 삭제하려면?
del color['cherry']
print(color)

color.clear()
print(color)

# 딕셔너리 값 수정, 추가, 삭제
device = {'아이폰':5, '아이패드':10, '윈도우타블렛':20}
device['맥프레']=15
device['아이폰']=6
print(device)
del device['아이폰']
print(device)

print(device.keys()) # 딕셔너리 키들의 목록
print(device.values())  # 딕셔너리 값들의 목록

# 딕셔너리를 for in 구문으로 참조하기
D = {'a':1, 'b':2, 'c':3}
for key in D.keys():
    print(key, D[key])
# 딕셔너리에 있는 키와 값을 출력할 경우 위와 같이 for 아이템 in 딕셔너리:
# 구문을 사용해서 루프를 돌면 아이템은 키를 D[아이템]은 키에 맵핑된 값을 출력함.

# 리스트와 딕셔너리의 차이점은 리스트는 순서가 있고 딕셔너리는 순서가 없다.
# 따라서 리스트는 정수의 인덱스를 가지고 있고 딕셔너리는 키로 값을 꺼내야 한다.