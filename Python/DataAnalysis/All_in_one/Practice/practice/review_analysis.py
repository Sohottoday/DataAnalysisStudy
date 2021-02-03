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
print("tf_idf_vect[0] : ", tf_idf_vect[0])

### 벡터 : 단어 맵핑
invert_index_vectorizer = {v : k for k,v in vect.vocabulary_.items()}
print("str(invert_index_vectorizer)[:100] : ", str(invert_index_vectorizer)[:100])


# Logistic Regression 분류
## 데이터셋 생성
print(df.sample(10).head())

def rating_to_label(rating):        # 평점이 3 이상인 값과 그렇지 않은값 분류
    if rating > 3:
        return 1
    else:
        return 0

df['y'] = df['rating'].apply(lambda x : rating_to_label(x))

print(df.y.value_counts())

## 데이터셋 분리
from sklearn.model_selection import train_test_split

y = df['y']
x_train, x_test, y_train, y_test = train_test_split(tf_idf_vect, y, test_size=0.3)

## 모델 학습
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

# score들을 출력해보면 수치가 뭔가 이상하다는 것을 알 수 있다.

from sklearn.metrics import confusion_matrix

confmat = confusion_matrix(y_test, y_pred)
print(confmat)
"""
[[  4  85]
 [  0 212]]
샘플링의 비율이 달라 confusion matrix값이 이상하다는 것을 알 수 있다.
1:1로 샘플링을 재조정 해준다.
"""

## 샘플링 재조정
positive_random_idx = df[df['y']==1].sample(275, random_state=33).index.tolist()
negative_random_idx = df[df['y']==0].sample(275, random_state=33).index.tolist()

random_idx = positive_random_idx + negative_random_idx
x = tf_idf_vect[random_idx]
y = df['y'][random_idx]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

confmat = confusion_matrix(y_test, y_pred)
print(confmat)


## 긍정/부정 키워드 분석
### LogisticRegression 모델의 coef 분석
plt.rcParams['figure.figsize'] = [10, 8]
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
plt.show()

### 긍정/부정 키워드 출력
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[-5:])

coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)      # 상위 n개의 긍정적 키워드
coef_neg_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=False)      # 상위 n개의 긍정적 키워드

invert_index_vectorizer = {v:k for k, v in vect.vocabulary_.items()}

for coef in coef_pos_index[:15]:        # 긍정적 단어 15개 출력
    print(invert_index_vectorizer[coef[1]], coef[0])

print("------------------------------------------")
for coef in coef_neg_index[:15]:        # 부정적 단어 15개 출력
    print(invert_index_vectorizer[coef[1]], coef[0])

