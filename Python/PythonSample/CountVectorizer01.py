from sklearn.feature_extraction.text import CountVectorizer     # feacture_extraction. ~ : ~~한 파일의 특징 뽑아내기

# CountVector : 문자열에서 단어 토큰을 생성하여 BOW로 인코딩된 벡터를 생성해 준다.
# df 의 f는 frequency(빈도 수)
# min_df=2 : 최소 빈도가 2번 이상인 단어들만 추출하라는 의미
# stop_words : 불용어

vectorizer = CountVectorizer(min_df=2, stop_words=['친구'])
print(type(vectorizer))

sentences = ['우리 아버지 여자 친구 이름은 홍길동 홍길동', '홍길동 여자 친구 이름은 심순애 심순애', '여자 친구 있나요.']

# 단어 사전
mat = vectorizer.fit(sentences)
print(type(mat))

print(mat.vocabulary_)
# {'여자': 0, '이름은': 1, '홍길동': 2}     이런식으로 답이 나오는데 그 이유는 한 문장에 같은 단어가 여러개 있어도 한번 나왔다고 취급한다.
# 그러므로 홍길동은 2번이라 표시되고 심순애는 한 문장에서만 2번 나왔으므로 최종적으로 1번 나온것이라 취급되어 min_df=2 에 의해 필터링된다.

print(sorted(mat.vocabulary_.items()))

features = vectorizer.get_feature_names()
print(type(features))
print(features)

print('불용어')
print(vectorizer.get_stop_words())

myword = [sentences[0]]
print('myword : ', myword)

myarray = vectorizer.transform(myword).toarray()
print(type(myarray))

'''
0('여자')이 1번, 1('이름은')이 1번, 2('홍길동')가 2번 나왔습니다.
myarray : [[1 1 2]]
단어 사전 : {'여자':0, '이름은': 1, '홍길동' : 2}
myword : ['우리 아버지 여자 친구 이름은 홍길동 홍길동'
'''

print('myarray : ',myarray)

