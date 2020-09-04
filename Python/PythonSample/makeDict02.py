
fruits = [('바나나', 10), ('수박', 20), ('오렌지', 30)]

mydict = dict()

for key, value in fruits:
    mydict[key] = value

print(mydict)

fruits = [('바나나', 10), ('수박', 20), ('오렌지', 30), ('바나나', 50)]

mydict1 = dict()
for key, value in fruits:
    if not key in mydict1:
        mydict1[key] = value
    else:
        imsi = mydict1[key]
        mydict1[key] = imsi + value


print(mydict1)