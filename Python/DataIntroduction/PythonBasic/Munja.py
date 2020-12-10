
a = '01-sample.png'
b = '02-sample.jpg'
c = '03-sample.pdf'

# startswith : ~로 시작하는        endswith : ~로 끝나는
a.startswith('01')          # True값을 리턴
a.endswith('.jpg')          # False값을 리턴

mylist = [a, b]
for file in mylist:
    if file.endswith('jpg'):
        print(file)

aa = a.replace('.png', 'jpg')
print(aa)       # 01-sample.jpg

# strip : 양 끝 공백 제거
d = '    01-sample.png'
print(a==d)     # false

print(d.strip() == a)