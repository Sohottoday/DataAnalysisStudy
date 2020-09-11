
from xml.etree.cElementTree import parse

# parse : xml 문서를 파싱해주는 함수
tree = parse('xmlEx_03.xml')
myroot = tree.getroot()
print(type(myroot))
print('-'* 30)

# 해당 속성들의 키를 list 형식으로 반환
print(myroot.keys())
print('-'* 30)


print(myroot.items())
print('-'* 30)

print(myroot.get('설명'))
print('-'* 30)

print(myroot.get('qwert', '없을 경우 기본값'))
print('-'* 30)

print(myroot.findall('가족'))
print('-'* 30)

family = myroot.find('가족')
print('-'* 30)

#childs = family.getchildren()
childs = [item for item in family]
#print(childs)

for onesaram in childs:
    #print(onesaram)
    #print('-'*20)
    elem = [item for item in onesaram]
    for abc in elem:
        print(abc.text)
        if abc.text == '이순자':
            print(abc.attrib['정보'])
    print('^'*30)

print('%' * 30)

allfamily = myroot.findall('가족')
for onefamily in allfamily:
    #families = [item for item in onefamily]
    for onesaram in onefamily:
        print(onesaram)
        name = onesaram.find('이름')
        if name != None:
            print(name.text)
        else:
            print(onesaram.attrib['이름'])





filename = 'xmlEx_03.xml'