# 토지 파일을 읽어 들여서 워드 클라우드와 막대 그래프 그리기
import numpy as np
from PIL import Image
from wordcloud import WordCloud     # 참고 : pytagcloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

class Visualization:
    def __init__(self, wordlist):
        self.wordlist = wordlist
        self.wordDict = dict(self.wordlist)

    def makeWordCloud(self):
        alice_color_file = 'alice_color.png'

        # 이미지를 numpy 배열로 바꿔준다.
        alice_coloring = np.array(Image.open(alice_color_file))

        fontpath = 'malgun.ttf'
        wordcloud = WordCloud(font_path=fontpath, mask=alice_coloring, background_color='lightyellow', relative_scaling=0.2)
        wordcloud = wordcloud.generate_from_frequencies(self.wordDict)

        image_colors = ImageColorGenerator(alice_coloring)
        # random_state : 랜덤 상수 지정
        newwc = wordcloud.recolor(color_func=image_colors, random_state=42)

        plt.imshow(newwc)
        plt.axis('off')

        filename = 'tojiWordCloud.png'
        plt.savefig(filename)
        plt.figure(figsize=(16, 8))



    def makeBarChart(self):
        # result를 이용하여 막대 그래프를 그려보시오
        result = self.wordlist[0:10]        # 10개 데이터
        xchuck = [list[0] for list in result]
        ychuck = [list[1] for list in result]
        #print(result)
        plt.bar(xchuck, ychuck)
        plt.show()

        '''
            강사님 코드
            plt.
        '''
        

filename = 'tojiText.txt'
ko_con_text = open(filename, 'rt', encoding='utf-8').read()

from konlpy.tag import Okt

okt = Okt()
token_ko = okt.nouns(ko_con_text)

# 불용어(stopword) : 빈도 수에 상관 없이 분석에서 배제할 단어들
stop_word_file = 'stopword.txt'
stop_file = open(stop_word_file, 'rt', encoding='utf-8')
stop_words = [word.strip() for word in stop_file.readlines()]

token_k = [each_word for each_word in token_ko if each_word not in stop_words]

# print(stop_words)
# nltk : national language toolkit : 자연어 처리를 위한 툴킷
# token : 작은 절편
import nltk
ko = nltk.Text(tokens=token_ko)

wordlist = list()   # 튜플(단어, 빈도수)를 저장할 리스트
# 가장 빈도수가 많은 500개만 추출
data = ko.vocab().most_common(500)
#print(data)
for word, count in data:
    if count >= 50 and len(word) >= 2:
        wordlist.append((word, count))

visual = Visualization(wordlist)
visual.makeWordCloud()
visual.makeBarChart()



print('finished')