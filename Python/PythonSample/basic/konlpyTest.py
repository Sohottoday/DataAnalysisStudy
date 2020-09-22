from konlpy.tag import Komoran

sentence = '국정 농단 태블릿 PC, 설진욱, 가나다라'
print('# before user dic')
komo = Komoran()
print(komo.pos(sentence))