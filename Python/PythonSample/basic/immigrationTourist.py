# 출입국 관광 통계 서비스
import urllib.parse
import urllib.request
import json
import matplotlib.pyplot as plt

# 한글 깨짐 방지
plt.rc('font', family='Malgun Gothic')

def getRequestUrl(url):     # 해당 url 문자열을 이용하여 정보를 읽어 옵니다.
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :      # 문제가 없다는 의미
            #print(f'url 정보 : [{url}]')
            #print(f'발생 시각 : [{datetime.datetime.now()}]')
            return response.read().decode('utf-8')

    except Exception as err:
        print(err)
        #print(f'발생 시각 : {datetime.datetime.now()}')
        print(f'오류 발생 : {url}')
        return None

access_key = 'FmwRWVnhOa51nyy7CnhK1MHQejORL0pJhWtMU6fgmhud66vbgwyNsBUhWLi%2FldT5ec82gHRTCPNG%2BOtLw4ZQwg%3D%3D'

def getNatVisitor(yyyymm, nat_cd, ed_cd):
    end_point = 'http://openapi.tour.go.kr/openapi/service'
    end_point+= '/EdrcntTourismStatsService'
    end_point += '/getEdrcntTourismStatsList'

    parameters = '?'
    parameters += '_type=json'
    parameters += '&serviceKey='+access_key
    parameters += '&YM=' + yyyymm        # 년월
    parameters += '&NAT_CD' + nat_cd     # 국가코드
    parameters += '&ED_CD' + ed_cd      # 출국/입국

    url = end_point + parameters

    retData = getRequestUrl(url)

    if retData == None:
        return None
    else:
        return json.loads(retData)

def main():
    jsonResult = []

    nation = '중국'       # 중국 : 112, 일본 : 130, 미국 : 275
    national_code = '112'
    cd_ed = 'E'     # 방한한 외국인 관광객

    nStartyear = 2015
    nEndYear = 2020

    for year in range(nStartyear, nEndYear+1):
        for month in range(1, 13) :
            yyyymm = '%s%s' % (str(year), str(month).zfill(2))

            jsonData = getNatVisitor(yyyymm, national_code, cd_ed)
            print(jsonData)

            if jsonData['response']['header']['resultMsg'] == 'OK':
                # krName = jsonData['response']['body']['items']['item']['natKorNm']
                # krName = krName.replace(' ', '')

                totalCount = jsonData['response']['body']['totalCount']
                if totalCount !=0 :
                    iTotalVisit = jsonData['response']['body']['items']['item']['num']
                    # print('%s_%s : %s' % (krName, yyyymm, iTotalVisit))

                    jsonResult.append({'nat_name':nation, 'nat_cd':national_code, 'yyyymm':yyyymm, 'visit_cnt':iTotalVisit})
                    print(a)




    # 파일 저장하기
    savedFilename = f'immigrationTourist {nation}({national_code})_({nStartyear}~{nEndYear}).json'

    with open(savedFilename, 'w', encoding='utf-8') as outfile:
        retJson = json.dumps(jsonResult, indent=4, ensure_ascii=False, sort_keys=True)
        outfile.write(retJson)
        
    print(savedFilename + ' 파일 저장됨')

    plt.xlabel('방문월')
    plt.ylabel('방문객수')
    plt.show()

if __name__ == '__main__':
    main()