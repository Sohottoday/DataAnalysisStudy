
a = 14
b = 5

sum = a+b
sub = a-b
multiply = a * b
divide = a / b
divide2 = a // b
remainder = a % b
power = 2 ** 10

print('덧셈 : %d' % (sum))
print('뺄셈 : %d' % (sub))
print('곱셈 : %d' % (multiply))
print('나눗셈 : %f' % (divide))
print('나눗셈2 : %d' % (divide2))
print('나머지 : %d' % (remainder))
print('제곱 : %d' % (power))

print('제곱수2 : [%3d]' % (power))     # 실제 자릿수보다 지정된 서식이 작으면 지정된 서식이 무시된다.
print('제곱수3 : [%6d]' % (power))     # 양의 정수는 우측정렬되며 자릿수를 채운 뒤 나머지는 빈 공백으로 채워진다.
print('제곱수4 : [%-6d]' % (power))    # 음의 정수는 좌측정렬되며 자릿수를 채운 뒤 나머지는 빈 공백으로 채워진다.

su = 12.3456789
print('서식1 : [%f]' % (su))
print('서식2 : [%.2f]' % (su))
print('서식3 : [%6.2f]' % (su))
print('서식4 : [%-6.2f]' % (su))