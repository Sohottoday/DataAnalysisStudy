import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/bourne_scenario.csv")
print(df.head())
"""
Feature Description
    page_no : 데이터가 위치한 pdf 페이지 정보
    scene_title : 씬 제목
    text : 씬에 해당하는 지문/대본 텍스트 정보
"""

# 데이터셋 살펴보기
print(df.shape)
print(df.info())


# 텍스트 데이터 전처리
## 정규표현식 적용
import re

def apply_regular_expression(text):     # 정규표현식을 실행하는 함수를 하나 만들어 준다.
    text = text.lower()     # 컴퓨터는 대소문자 구분을 따로하므로 모두 소문자로 변환
    english = re.compile('[^ a-z]')     # ^ a-z는 a-z 즉, 영어가 아닌 나머지 특수문자, 숫자 등을 매칭한다는 의미
    result = english.sub('', text)      # compile을 통해 elgilsh가 영어를 제외한것들을 매칭했기 때문에 영어를 제외하고 모두 '', 즉, 삭제하라는 의미
    result = re.sub(' +', ' ', result)          # ' +' 의 의미는 하나 이상의 공백 이라는 의미. 즉, 하나 이상의 공백이 있을 경우 하나의 공백으로 변환하라는 의미
    return result

print(apply_regular_expression(df['text'][0]))

df['preprocessed_text'] = df['text'].apply(lambda x : apply_regular_expression(x))
print(df.head())

# Word Count

## 말뭉치(코퍼스) 생성
corpus = df['preprocessed_text'].tolist()       # 텍스트 데이터를 리스트형식으로 뭉탱이로 가져옴

## BoW 벡터 생성 : Back of Words
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(tokenizer=None, stop_words="english", analyzer='word').fit(corpus)
# tokenizer : 정규표현식과 같이 들어오는 문장들을 어떻게 설정할건지. 만약 한국어 문서라고 가정한다면 한국어 정규표현식을 설정한 함수를 넣어주면 된다.
# stop_words : 불용어. 실질적으로 의미를 가지고 있지 않은 단어들을 제거한다는 의미
# analyzer : 분석 단위. word이므로 단어단위로 분석
bow_vect = vect.fit_transform(corpus)
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)

print(word_list[:5])
print(count_list)           # word_list들이 몇번 등장했는지
print(bow_vect.shape)
print(bow_vect.toarray())
print(bow_vect.toarray().sum(axis=0))       # 열을 기준으로 다 더함. 즉, 몇번 등장했는지 확인 가능

word_count_dict = dict(zip(word_list, count_list))      # 등장하는 단어와 횟수를 딕셔너리 형식으로 묶어준다.
print(str(word_count_dict)[:100])

import operator

sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)       # 빈도수가 높은 순서대로 정렬해본다.
print(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:5])

## 단어 분포 탐색
plt.hist(list(word_count_dict.values()), bins=150)
plt.show()


# 텍스트 마이닝
## 단어별 빈호 분석
### 워드클라우드 시각화
#### pip install pytagcloud pygame simplejson

from collections import Counter
import random
import pytagcloud
import webbrowser

ranked_tags = Counter(word_count_dict).most_common(25)          # 가장 많이 등장한 단어 25개를 출력하라는 의미
print(ranked_tags)

taglist = pytagcloud.make_tags(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:40], maxsize=60)
# 정렬하여 40개를 출력하는것과 위의 most_common(40)으로 준 부분은 같은 의미이다.
pytagcloud.create_tag_image(taglist, 'wordcloud_example.jpg', rectangular=False)

from IPython.display import Image
Image(filename='wordcloud_example.jpg')


## 장면별 중요 단어 시각화
### TF-IDF 변환 : 위에서의 단순한 텍스트 마이닝은 의미가 없는 단어들까지도 마이닝된다.
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
print(tf_idf_vect.shape)        # 사이즈와 단어 총 개수를 알 수 있다.
print(tf_idf_vect[0])
print(tf_idf_vect[0].toarray().shape)       # toarray()를 통해 보기 쉽게 만들어본다.
print(tf_idf_vect[0].toarray())
"""
  (0, 2788)     0.19578974958217082
  (0, 2763)     0.27550455848587985
  (0, 2412)     0.1838379942679887
  (0, 2387)     0.3109660261831164
  (0, 1984)     0.2902223973596984
  (0, 1978)     0.3109660261831164
  (0, 1898)     0.27550455848587985
  (0, 1673)     0.2902223973596984
  (0, 1366)     0.21520447034992146
  (0, 1251)     0.19855583314180728
  (0, 1001)     0.2340173008390438
  (0, 974)      0.2902223973596984
  (0, 874)      0.27550455848587985
  (0, 798)      0.1906694714764746
  (0, 237)      0.08646242181596513
  (0, 125)      0.26408851574819875
(1, 2850)
[[0. 0. 0. ... 0. 0. 0.]]
이런식의 결과가 나오는데 위의 값은 tf_idf_vect[0]의 값이 총 2850개의 단어 중 몇번째에 위치해 있는지를 확인할 수 있다.
옆의 소수점자리 숫자가 tf_idf 값이라 보면 된다. 수치가 높을수록 중요한 단어라는 의미
자주 등장하는 단어일수록 수치가 낮음.
"""

### 벡터 : 단어 맵핑
invert_index_vectorizer = {v : k for k, v in vect.vocabulary_.items()}      # 아래 주석에 대한 내용을 dict 형태로 저장
# print(vect.vocabulary_)
"""
{'raining': 1898, 'light': 1366, 'strobes': 2387, 'wet': 2763, 'glass': 1001, 'rhythmic': 1978, 'pace': 1673, 'suddenly': 2412, 'window': 2788, 'face': 798, 'jason': 1251, 
'bourne': 237, 'riding': 1984, 'backseat': 125, 'gaze': 974, 'fixed': 874, 'knee': 1297, 'syringe': 2459, 'gun': 1055, 'eyes': 795, 'driver': 703, 'jarda': 1248, 'watching': 2741, 'bournes': 240
이런식으로 값이 출력되는데 해당하는 당언가 몇번째에 위치해 있는지를 나타내는 것이다.
ex) raining은 1898번째에 위치
"""

### 중요 단어 추출 - Top 3 TF-IDF
np.argsort(tf_idf_vect.toarray())[:, -3:]
print(np.argsort(tf_idf_vect.toarray())[:, -3:][:5])
"""
[[1984 2387 1978]
 [1297 1971 1097]
 [1693 2221  968]
 [ 690  299 1482]
 [2823 1951 1454]]
이런식의 값이 shape에서 확인한것처럼 320개가 나오는데
320개의 문장들에서 가장 중요한 단어(tf-idf값이 높은 단어)들 top3 2 1 을 나타낸 것이다.
즉, 맨 위 값은 첫 번째 문장에서 가장 중요한 단어 3 : 1983, 2 : 2387, 1 : 1978
"""

## df로 전환하여 실제 단어로 전환
top_3_word = np.argsort(tf_idf_vect.toarray())[:, -3:]
df['important_word_indexes'] = pd.Series(top_3_word.tolist())
print(df.head())

def convert_to_word(x):
    word_list = []
    for word in x:
        word_list.append(invert_index_vectorizer[word])
    return word_list

df['important_words'] = df['important_word_indexes'].apply(lambda x : convert_to_word(x))
print(df.head())