# 학생의 이름과 국어, 영어, 수학 점수를 입력 받으세요.
# 김철수, 50, 60, 80
# 총점은 소수점 2자리로, 평균은 소수점 3자리로 출력하세요.
# 출력 결과
# 이름 : 김철수
# 국어 : 50점
# 영어 : 60점
# 수학 : 80점
# 총점 : 190.00
# 평균 : 63.333

name = input('이름 : ')
kor = int(input('국어 : '))
eng = int(input('영어 : '))
math = int(input('수학 : '))

print("""
이름 : %s
국어 : %d점
영어 : %d점
수학 : %d점
총점 : %.2f
평균 : %.3f
""" % (name, kor, eng, math, (kor+eng+math), ((kor+eng+math)/3)))
