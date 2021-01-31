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
