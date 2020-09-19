
import numpy as np
import matplotlib.pyplot as plt

def some_function1(h):
    return -np.log(h)

def some_function2(h):
    return -np.log(1-h)

x1 = np.arange(1e-8, 0.51, 0.01)        # 로그 함수 특성상 0은 무한대이므로 1e-8 로 표시
y1 = some_function1(x1)

x2 = np.arange(0.5, 1.0, 0.01)
y2 = some_function2(x2)

plt.plot(x1, y1, marker='', color='r', label='-nplog(h)')
plt.plot(x2, y2, marker='', color='g', label='-np.log(1-h)')

plt.grid(True)
plt.xlim(-0.2, 1.2)
plt.ylim(0.0, 5.0)
plt.xlabel('weight')
plt.ylabel('cost')
plt.title('logistic cost function')

plt.savefig('logistic_function.png')