# KoNLP를 이용한 형태소 분석
## KoNLP가 제공하는 형태소 분석기 중 하나인 Kkma 사용
from konlpy.tag import Hannanum
from konlpy.tag import Twitter
from konlpy.tag import Kkma

hannanum = Hannanum()
twitter = Twitter()
kkma = Kkma()

# 형태소 분석기
"""
- 한나눔 : http://semanticweb.kaist.ac.kr/hannanum/index.html
- 트위터 : https://github.com/twitter/twitter-korean-text
- 꼬꼬마 : http://kkma.snu.ac.kr/documents/
"""

# 꼬꼬마 형태소 분석기
"""
문장을 형태소 단위로 분리하고 품사를 태깅한다.
품사태그는 일반명사(NNG), 고유명사(NNP), 동사(VV), 형용사(VA) 등이 있다.
http://kkma.snu.ac.kr/documents/index.jsp?doc=postag 형태소 리스트 확인
"""

print(kkma.sentences(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))      # sentences로 나눠줌(문장단위로)
print(kkma.pos(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))        # 각각 태깅을 해준다.
print('---' * 30)
# 한나눔 형태소 분석기
print(hannanum.pos(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))
print('---' * 30)
# 트위터 형태소 분석기
print(twitter.pos('아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))
print('---' * 30)


# 형태소 분석 - kkma 명사
line_list = []
f = open('centrum_review.txt', encoding='utf-8')
for line in f:
    line = kkma.nouns(line)
    line_list.append(line)
f.close()

print("- 불러온 문서 : ", len(line_list), "문장")

print(line_list[10])

# 단어 빈도 체크
## 일반적으로 명사로 분석을 하고 추가 분석이 필요할 경우 형용사나 타른 품사를 분석한다.
word_frequency = {}
noun_list = []
# 불용어 리스트를 여기에 추가한다.
stop_list = ["배달"]          # 이 대괄호 안에 불용단어를 넣어주면 된다.
line_number = 0
for line in line_list[:]:
    line_number += 1
    print(str(line_number) + "/" + str(len(line_list)), end='\r')
    noun = []
    for word in line:
        if word.split('/')[0] not in stop_list and len(word.split('/')[0]) > 1:
            noun.append(word.split('/')[0])
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1
        noun_list.extend(noun)

# 단어별 출현빈도를 출력
word_count = []
for n, freq in word_frequency.items():
    word_count.append([n, freq])
word_count.sort(key=lambda elem : elem[1], reverse=True)
for n, freq in word_count[:10]:
    print(n + '\t' + str(freq))

# 추출한 명사 리스트를 활용해 명사만으로 이루어진 문서 생성
noun_doc = ' '.join(noun_list)
noun_doc = noun_doc.strip()

# 워드클라우드 시각화
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

from matplotlib import rc       # 왜 한글이 깨지는가?
rc('font', family = 'Malgun Gothic')

#워드클라우드 파라미터 설정
# font_path="Malgun Gothic"  #폰트
background_color="white"      #배경색
margin=3                     #모서리 여백 넓이
min_font_size=7              #최소 글자 크기
max_font_size=150             #최대 글자 크기
width=500                     #이미지 가로 크기
height=500                    #이미지 세로 크기
wc = WordCloud( background_color=background_color, margin=margin, \
               min_font_size=min_font_size, max_font_size=max_font_size, width=width, height=height)        # font_path=font_path,
wc.generate(noun_doc)

plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()          # 자꾸 한글 폰트가 깨져서 출력됨


# LDA 토픽 모델링
import gensim
from gensim import corpora
import logging

logging.basicConfig(level=logging.DEBUG)
topic = 5               # 문서가 있을 때 몇 개의 토픽으로 구성되어 있는지 미리 셋팅하는 것
keyword = 10            # 몇 개의 키워드로 구성되어 있는지 미리 셋팅하는 것
texts = []
resultList = []
stop_list = ['배송', '만족', '카페', '카페규정', '확인', '주수', '센트']        # 불용어
for line in line_list:
    words = line
    if words != ['']:
        tokens = [word for word in words if (len(word.split('/')[0]) > 1 and word.split('/')[0] not in stop_list)]
        texts.append(tokens)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic, id2word=dictionary, passes=10)
for num in range(topic):
    resultList.append(ldamodel.show_topic(num, keyword))

print('---' * 20)
print(resultList)
"""
[[('구매', 0.03883506), ('번째', 0.03882187), ('선물', 0.021939773), ('비타민', 0.014656154), ('번창', 0.014613918), ('제품', 0.014600864), ('적극추천', 0.014571322), ('적 
극', 0.014562527), ('아빠', 0.014561847), ('기분', 0.0145607665)], [('주문', 0.026903996), ('구매', 0.026874859), ('건강', 0.026858771), ('가격', 0.026834935), ('센트룸', 0.0268076), ('이용', 0.026798466), ('아들', 0.026784051), ('군대간', 0.026770342), ('포맨', 0.026770175), ('대간', 0.026770145)], [('감사', 0.09384217), ('가격', 0.039471757), ('포장', 0.03919923), ('피곤', 0.021527365), ('부모님', 0.021508299), ('피곤하다', 0.021498721), ('얼마안됬지', 0.02149845), ('하다', 0.021498326), ('용량', 0.021498073), ('가격대비', 0.02149749)], [('추천', 0.04524783), ('구매', 0.031119108), ('효과', 0.031079939), ('남편', 0.031076793), ('적극', 0.031071993), ('도착', 0.031071898), ('비 
타민', 0.031071033), ('제품', 0.031021805), ('감사', 0.030745031), ('복용', 0.017013941)], [('비타민', 0.04252993), ('건강', 0.04252463), ('센트룸', 0.02925072), ('영양소', 0.029155204), ('포장', 0.01600448), ('상품', 0.015985703), ('피곤', 0.015968768), ('기준', 0.015967371), ('노란색', 0.01596736), ('배출', 0.015967354)]]

토픽을 5개 키워드를 10개로 값을 줬을때 위와 같은 방식으로 출력되는데 분석하는 사람은 이러한 결과를 보고 topic 값과 keyword값을 조정하면서 분석 결과값을 도출하면 된다.
"""

"""
위와 같이 토픽 모델을 분류하여 다이어그램을 만들어 기획 혹은 마케팅 등에 사용하면 된다.
ex) 네트워크 다이어그램
추천프로그램 : gephi
"""