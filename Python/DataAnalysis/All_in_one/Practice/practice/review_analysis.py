import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/tripadviser_review.csv")
print(df.head())

print(df.shape)
print(df.isnull().sum())
print(df.info())
print(len(df['text'].values.sum()))     # 데이터 전체에 등장하는 문자열의 갯수

# 한국어 텍스트 데이터 전처리
## konlpy 설치
## pip install konlpy

## 정규 표현식 적용
import re

def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')
    result = hangul.sub('', text)
    return result

print(apply_regular_expression(df['text'][0]))

## 한국어 형태소 분석 - 명사 단위
### 명사 형태소 추출
from konlpy.tag import Okt
from collections import Counter

nouns_tagger = Okt()
nouns = nouns_tagger.nouns(apply_regular_expression(df['text'][0]))       # Okt에는 코퍼스 즉, 말뭉치가 들어가야 한다.
print(nouns)

nouns = nouns_tagger.nouns(apply_regular_expression("".join(df['text'].tolist())))      # join을 통해 많은 문장들을 하나의 말뭉치로 다 합쳐준다.(Okt에는 코퍼스만 들어가야 하므로)

counter = Counter(nouns)
print(counter.most_common(10))      # 이 결과 한글자인 것은 의미가 없는 단어가 많으므로 2글자 이상의 단어만 나올 수 있도록 필터링 해준다.

available_counter = Counter({x : counter[x] for x in counter if len(x) > 1})
print(available_counter.most_common(10))

## 불용어 사전
stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
print("임시 불용어 사전 : ", stopwords[:10])
# 이 외에도 자신이 분석하려는 분야에 있어서 의미가 없을듯한 단어들은 직접 불용어를 지정해준다.
jeju_list = ['제주', '제주도', '호텔', '리뷰', '숙소', '여행', '트립']
for word in jeju_list:
    stopwords.append(word)

## Word Count
### Bow 벡터 생성
from sklearn.feature_extraction.text import CountVectorizer

def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')
    result = hangul.sub('', text)
    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(result)
    nouns = [x for x in nouns if len(x) > 1]
    nouns = [x for x in nouns if x not in stopwords]
    return nouns

vect = CountVectorizer(tokenizer= lambda x : text_cleaning(x))

bow_vect = vect.fit_transform(df['text'].tolist())
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)

print(word_list[:10])
print(count_list)
print(bow_vect.shape)

word_count_dict = dict(zip(word_list, count_list))
print(str(word_count_dict)[:100])

## TF-IDF 적용
### TF-IDF 변환
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
print(tf_idf_vect[0])

### 벡터 : 단어 맵핑
invert_index_vectorizer = {v : k for k,v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100])
