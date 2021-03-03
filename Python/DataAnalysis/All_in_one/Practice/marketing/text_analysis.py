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
background_color="white"      #배경색
margin=3                     #모서리 여백 넓이
min_font_size=7              #최소 글자 크기
max_font_size=150             #최대 글자 크기
width=500                     #이미지 가로 크기
height=500                    #이미지 세로 크기
wc = WordCloud(background_color=background_color, margin=margin, \
               min_font_size=min_font_size, max_font_size=max_font_size, width=width, height=height)
wc.generate(noun_doc)

plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()          # 자꾸 한글 폰트가 깨져서 출력됨