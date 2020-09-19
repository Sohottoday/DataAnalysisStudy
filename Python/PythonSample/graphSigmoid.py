
import numpy as np
import matplotlib.pyplot as plt

#plt.rc('font', family='Malgun Gothic')

def sigmoid(weight, x, b=0, asc=True):
    if asc == True:
        return 1/(1 + np.exp(-weight * x - b))
    else:
        return 1/(1 + np.exp(+weight * x - b))

x = np.arange(-5.0, 5.1, 0.1)

weight, bias = 1, 0
y1 = sigmoid(weight, x)
mylabel = f'y={str(weight)}*x{str(bias)}'
plt.plot(x, y1, color='g', label=mylabel)

weight, bias = 5, 0
y2 = sigmoid(weight, x, bias)
mylabel = f'y={str(weight)}*x{str(bias)}'
plt.plot(x, y2, color='b', label=mylabel)

weight, bias = 5, 3
y3 = sigmoid(weight, x, bias)
mylabel = f'y={str(weight)}*x{str(bias)}'
plt.plot(x, y3, color='r', label=mylabel)

weight, bias = 5, 3
y3 = sigmoid(weight, x, asc=False)
mylabel = f'y={str(weight)}*x{str(bias)}'
plt.plot(x, y3, color='r', label=mylabel)

plt.axhline(y=0, color='black', linewidth=1, linestyle='dashed')
plt.axhline(y=1, color='black', linewidth=1, linestyle='dashed')

plt.title('sigmoid function')
plt.ylim(-0.1, 1.1)
plt.legend(loc='best')      # 범례

plt.savefig('sigmoid_function.png')


