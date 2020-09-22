import json

filename = 'some.json'

myfile = open(filename, 'rt', encoding='utf-8')

myfile = myfile.read()

print(myfile)
print(type(myfile))

some = json.loads(myfile)

mem = some['member']
for i, n in mem.items():
    print(f'{i}는 {n}입니다.')

web = some['web']
for i, n in web.items():
    print(f'{i}는 {n}입니다.')


