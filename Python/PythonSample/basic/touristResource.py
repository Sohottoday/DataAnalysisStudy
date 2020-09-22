import urllib.parse
import urllib.request
import json
import math
import datetime

'''
    처리 순서
    1. end_point 문자열 생성
    2. 일반 인증 키를 발급 받기
    3. 요청을 하기 위한 url 변수 생성

'''

print(datetime.datetime.now())

# 공공 기관 데이터 : 관광 자원 통계 서비스

# 인증 키
access_key = 'yKotFlFGptL%2FsL33UiS6F03eSNNLKQ%2BideRqqkDfBN0JTP3lVA3DIwk6Gn55eUEuZhvAVeIsmRusPpfBxET0Qg%3D%3D'

def getRequestUrl(url):     # 해당 url 문자열을 이용하여 정보를 읽어 옵니다.
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :      # 문제가 없다는 의미
            print(f'url 정보 : [{url}]')
            print(f'발생 시각 : [{datetime.datetime.now()}]')
            return response.read().decode('utf-8')

    except Exception as err:
        print(err)
        print(f'발생 시각 : {datetime.datetime.now()}')
        print(f'오류 발생 : {url}')
        return None

# end_point = 'http://openapi.tour.go.kr/openapi/service'
# end_point +='/TourismResourceStatsService'
# end_point +='/getPchrgTrrsrtVisitorList'

# parameters = '?'
# parameters += '_type=json'      #json으로 받겠다는 의미

# parameters += '&servicekey=' + access_key

# parameters += '&YM=' + '201610'
# parameters += '&SIDO=' + urllib.parse.quote('서울특별시')
# parameters += '&GUNGU=' + urllib.parse.quote('종로')
# parameters += '&RES_NM=' + urllib.parse.quote('경복궁')
# parameters += '' + 'a'
# parameters += '' + 'a'

# url = end_point + parameters
#print(url)

# result = getRequestUrl(url)
# print(result)

def getTourData(yyyymm, sido, gungu, nPageNum, maxRecords):
    # end_point와 parameters를 이용하여 url을 생성한다.
    # getRequestUrl() 함수를 호출하여 url이 반환하는 정보를 추출한다.
    end_point = 'http://openapi.tour.go.kr/openapi/service'
    end_point += '/TourismResourceStatsService'
    end_point += '/getPchrgTrrsrtVisitorList'

    parameters = '?'
    parameters += '_type=json'  # json으로 받겠다는 의미

    parameters += '&serviceKey=' + access_key

    parameters += '&YM=' + yyyymm
    parameters += '&SIDO=' + urllib.parse.quote(sido)
    parameters += '&GUNGU=' + urllib.parse.quote(gungu)
    parameters += '&RES_NM=' + ''
    parameters += '&pageNo=' + str(nPageNum)
    parameters += '&numOfRows=' + str(maxRecords)

    url = end_point + parameters

    result = getRequestUrl(url)

    if result == None:
        return None
    else:
        return json.loads(result)
# end def getTourData


def TourPointCorrection(item, yyyymm, jsonResult):
    # 전처리 : 해당 키가 존재하지 않는 경우, 기본 값으로 대체 데이터를 만들어 주는 함수
    # item 키 중에 'addrCd'이 존재하면 그대로 사용, 없으면 0으로 대체
    addrCd = 0 if 'addrCd' not in item.keys() else item['addrCd']
    gungu = 0 if 'gungu' not in item.keys() else item['gungu']
    sido = 0 if 'sido' not in item.keys() else item['sido']
    resNm = 0 if 'resNm' not in item.keys() else item['resNm']
    rnum = 0 if 'rnum' not in item.keys() else item['rnum']
    ForNum = 0 if 'csForCnt' not in item.keys() else item['csForCnt']
    NatNum = 0 if 'csNatCnt' not in item.keys() else item['csNatCnt']

    jsonResult.append({'yyyymm':yyyymm, 'addrCd':addrCd, 'gungu':gungu, 'sido':sido, 'resNm':resNm, 'rnum':rnum, 'ForNum':ForNum, 'NatNum':NatNum})

def main():
    jsonResult = []         # 전체 목록을 저장할 변수
    sido = '서울특별시'
    gungu = ''
    nStarYear = 2015        # 검색 시작 년도
    nEndYear = 2019         # 검색 종료 년도
    nPageNum = 1            # 페이지 번호
    maxRecords = 100        # 조회될 행의 최대 수

    for year in range(nStarYear, nEndYear+1):
        for month in range(1, 13):
            #yyyymm = '%s%s' % (str(year), str(month).zfill(2))
            yyyymm = f'{str(year)}{str(month).zfill(2)}'
            #print(yyyymm)

            while(True):
                jsonData = getTourData(yyyymm, sido, gungu, nPageNum, maxRecords)
                #print(jsonData)

                # 응답 결과가 'OK'이면
                if jsonData['response']['header']['resultMsg'] == 'OK':
                    nTotal = jsonData['response']['body']['totalCount']
                    if nTotal == 0:
                        break

                    for item in jsonData['response']['body']['items']['item']:
                        TourPointCorrection(item, yyyymm, jsonResult)

                    nPage = math.ceil(nTotal / 100)     # ceil : 올림 한다는 의미
                    if(nPageNum == nPage):
                        break       # 마지막 페이지 입니다.

                    nPageNum += 1

                else:
                    break
                #break
            #break
        #break

    # 파일 저장하기
    # ex) touristResource(서울특별시 2015 ~ 2019).json
    savedFilename =  f'touristResource({sido}, {nStarYear}, {nEndYear}).json'
    # savedFilename = 'touristResource(%s %d~%d).json' % (sido, nStarYear, nEndYear)
    with open(savedFilename, 'w', encoding='utf-8') as outfile:
        retJson = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
        outfile.write(retJson)
        
    print(savedFilename + '파일 저장됨')

if __name__ == '__main__':
    main()

print('-')
