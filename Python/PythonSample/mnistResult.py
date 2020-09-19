import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

test_acc = [0.9237, 0.9223, 0.9709, 0.9732, 0.9765]
test_loss = [0.2805, 0.2786, 0.0986, 0.1353, 0.0887]
comments = ['테스트01', '테스트02', '테스트03', '테스트04', '테스트05']

mycolor = ['b', 'g', 'r', 'c', 'b']

plt.figure()
plt.title('테스트 케이스별 정확도')
plt.xlabel('테스트 케이스')
plt.ylabel('정확도')
plt.bar(comments, test_acc, color=mycolor)

plt.savefig('mnist accuracy graph.png')

plt.figure()
plt.title('테스트 케이스별 비용(손실)함수')
plt.xlabel('테스트 케이스')
plt.ylabel('비용(손실)함수')
plt.bar(comments, test_loss, color=mycolor)

plt.savefig('mnist loss graph.png')