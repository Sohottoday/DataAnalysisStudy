
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement

note = Element('note')

SubElement(note, 'to').text = 'Tove'
SubElement(note, 'from').text = 'Jani'
SubElement(note, 'heading').text = 'Reminder'
SubElement(note, 'body').text = "Don't forget me this weekend!"

def indent(elem, level = 0):
    mytab = '\t'
    i = '\n' + level * mytab

    if len(elem) :
        if not elem.text or not elem.text.strip() :
            elem.text = i + mytab

        if not elem.tail or not elem.tail.strip() :
            elem.tail = i

        for elem in elem :
            indent(elem, level + 1)

        if not elem.tail or not elem.tail.strip() :
            elem.tail = i
    else :
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

indent(note)

xmlFile = 'xmlTest_02.xml'

ElementTree(note).write(xmlFile, encoding='utf-8')
print(xmlFile, ' 파일이 저장되었습니다.')