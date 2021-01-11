import numpy as np
from sklearn.linear_model import LinearRegression

x = np.arange(10).reshape(-1, 1)
y = (2*x + 1).reshape(-1, 1)        # reshape는 배열의 구조를 지정해줌 ex) 3x4  2x3 등. reshape에 -1의 의미는 지정해준 행 또는 열에 맞게 알아서 지정하라는 의미

# 모델 선언
model = LinearRegression()
print(model)

# 모델 학습
model.fit(x, y)

# 예측해보기 - 10을 넣었을 때 어떤 값이 나오는가? => 21이 나오면 정답
prediction = model.predict([[10.0]])
print(prediction)

"""
# X
- features라고 부른다.
- x_train, x_test
- 학습을 위한 데이터 세트. 예측할 값은 빠져있다.
- ex) 지역, 평형 정보, 층수 정보, 동네, 거주민 평균 나이 등

# Y
- labels라고 부른다.
- y_train, y_test
- 예측해야 할 값. 예측값만 존재한다.
- ex) 집 값

# train, test
- train
    모델이 학습하기 위해 필요한 데이터
    feature/label이 모두 존재
- test
    모델이 예측하기 위한 데이터
    feature만 존재

- 검증 데이터(적합도 검사 - 과대적합, 과소적합을 대바하기 위해)
- 학습을 위한 데이터 8 : 검증을 위한 데이터 2 => 보통 8:2 비율로 많이 진행한다.
    train data로 학습 / validation data로 모니터
- 절대 학습할 때 Validation Set가 관여되면 안된다. **

"""