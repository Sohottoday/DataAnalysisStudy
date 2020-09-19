# 신경망(Neural Net)은 크게 2가지로 나누어 진다.
# CNN(Convolution NN) : 이미지 처리
# RNN(Recurrent NN) : 순환 신경망, 시계열 데이터(주식 시세, 집값)
# 자연어 처리 :

# 문자열을 숫자화 : 단어 수준의 원핫 인코딩
sentences = ['tensorflow makes machine learning easy', 'machine learning is easy']

# 단어 사전(BOW : bag of words)
word_dict = {}

for onedata in sentences:
    for word in onedata.split():
        if word not in word_dict:
            # 번호 0은 내부 처리 용도
            word_dict[word] = len(word_dict) + 1        # 보통 자연어처리에서는 0번부터 시작이 아닌 1번부터 시작

print(len(word_dict))
print('-' * 30)

print(word_dict)
print('-' * 30)

max_length = 10         # 임의로 10으로 잡는 이유는 예를들어 단어가 매우 많을 때 많은 갯수 순으로 11위 이후부터는 자르겠다는 의미

sentences_length = len(sentences)
dict_size = max(word_dict.values()) + 1     # 사전의 크기

import numpy as np

results = np.zeros((sentences_length, max_length, dict_size))

print(results.shape)
print(results)
print('-' * 30)

for ii, sample in enumerate(sentences):
    for kk, word in list(enumerate(sample.split())):
        index = word_dict.get(word)
        results[ii, kk, index] = 1.
    print('#' * 30)

print(results)
print('-' * 30)

