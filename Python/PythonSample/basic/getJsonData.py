import json



def get_Json_Data():
    print('함수 호출됨')
    filename = 'jumsu.json'

    myfile = open(filename, 'rt', encoding='utf-8')
    print(type(myfile))

    myfile = myfile.read()
    print(type(myfile))     # 만약 외부에서 json 파일을 받았다면 이와같은 형식으로 str 형식으로 바꿔줘야한다.

    # loads(str) : 문자열 형식의 데이터를 이용하여 json 타입으로 변환해주는 함수
    jsonData = json.loads(myfile)
    print(type(jsonData))

    for oneitem in jsonData:
        print(oneitem.keys())
        print(oneitem.values())
        print('이름 : ', oneitem['name'])

        kor = float(oneitem['kor'])
        eng = float(oneitem['eng'])
        math = float(oneitem['math'])

        total = kor + eng + math
        print('총점 : ', total)

        if 'hello' in oneitem.keys():
            message = oneitem['hello']
            print('message : ', message)

        _gender = oneitem['gender'].upper()
        
        if _gender == 'M':      # _(언더바)를 붙이면 임시 변수라는 의미
            print('성별 : 남자')
        elif _gender == 'F':
            print('성별 : 여자')
        else:
            print('미정')

# __xx__ 이런식으로 언더바 2개가 있는 것은 파이썬이 내장하고 관리하는 내부 변수이다.
# 어플리케이션 이름이 저장되어 있다.
# 해당 어플리케이션이 스스로 실행되면 '__main__' 값이 대입된다.
if __name__ == '__main__':
    print('나 스스로 실행되었습니다.')
    get_Json_Data() # 함수 호출
else:
    print('다른 프로그램이 호출됐습니다.')
