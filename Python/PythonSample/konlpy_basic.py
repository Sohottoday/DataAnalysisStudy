
from konlpy.tag import Komoran

text = '세일즈 우먼인 아름다운 그녀가 아버지 가방에 들어 가시나 ㅎㅎ'

komo = Komoran()
print(type(komo))

# pos : 단어와 품사를 튜플로 만들어 리스트 형태로 반환
result = komo.pos(text)
print(result)

# 형태소 : 의미있는 최소의 단위
# 텍스트 마이닝 : 수많은 텍스트에서 의미있는 데이터만 추려 내는 행위

print('전체 확인하기')
for myitem in result:
    somedata = f'단어 : {myitem[0]}, 품사 : {myitem[1]}'
    print(somedata)
print('-' * 30)

print('2글자 이상의 명사만 추출해보기')
only_nouns = []
for myitem in result:
    if (myitem[1] in ['NNG', 'NNP']):
        if(len(myitem[0]) >= 3):
            somedata = f'단어 : {myitem[0]}, 품사 : {myitem[1]}'
            only_nouns.append(myitem[0])
        else:
            print(myitem[0] + '은 제외합니다.')


print(only_nouns)
#############
print('명사만 추출해보기')
nouns = komo.nouns(text)
print(nouns)
#############
# userdic 매개 변수 : 사용자 정의 사전을 활용하는 방법
sentence = '국정 농단 태블릿 PC, 최성연, 가나다라'
komo = Komoran(userdic='user_dic.txt')
print(komo.pos(sentence))

print('-' * 30)