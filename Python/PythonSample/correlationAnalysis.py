import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

plt.rc('font', family='Malgun Gothic')

corr_list = []  # 상관 관계 분석 결과를 저장
mychart = './mychart/'      # 그래프가 생성될 폴더

def correlation(x, y):      # 상관계수를 구해주는 함수
    bar_x = x.mean()
    bar_y = y.mean()

    bunja = np.sum((x-bar_x)*(y-bar_y))
    print('분자 : ', bunja)

    left = np.sum((x-bar_x)**2)
    right = np.sum((y - bar_y) ** 2)
    bunmo = np.sqrt(left * right)
    print('분모 : ', bunmo)

    return bunja / bunmo

def setScatterGraph(tour_table, visit_table, tourpoint):
    # tour_table : 관광지 입장 정보
    # visit_table : 3개국 관광객 수
    # tourpoint : 관광지 이름(예시 : 경복궁)

    tour = tour_table[tour_table['resNm'] == tourpoint]
    merge_table = pd.merge(tour, visit_table, left_index=True, right_index=True)
    #print('#' * 30)
    #print(merge_table)
    
    mylist = [['china', '중국인'], ['usa', '미국인'], ['japan', '일본인']]
    imsi = []
    imsi.append((tourpoint))

    fig = plt.figure()  # 도화지 준비

    print('[' + tourpoint + '] 작업 중입니다.')
    for onedata in mylist:
        plt.xlabel(onedata[1] + ' 입국수"')    # 예시) 중국인 입국수
        plt.ylabel('외국인 입장객수')

        # 해당 국가의 컬럼만 추출
        x_data = list(merge_table[onedata[0]])
        y_data = list(merge_table['ForNum'])

        cor = correlation(np.array(x_data), np.array(y_data))
        cor = round(cor, 6)

        if cor == 0:
            print('상관 계수가 0입니다.')
            return

        plt.title(tourpoint + '-' + onedata[1] + '상관 관계 분석(' + str(cor) + ')')
        plt.scatter(x_data, y_data, edgecolors='none', alpha=0.75, s=6, c='black')

        # savefig : 해당 이미지를 파일 형식으로 저장
        plt.savefig(mychart + tourpoint + '(' + onedata[1] + ').png')

        mytuple = (onedata[1], cor)
        imsi.append((mytuple))

    corr_list.append(imsi)



    
def main():
    if not os.path.exists(mychart):
        os.mkdir(mychart)

    # .은 현재 폴더를 의미한다.
    # ..는 상위 폴더를 의미한다.
    filename = './data/touristResourceStat(서울특별시 2015~2019).json'
    jsonTP = json.loads(open(filename, 'rt', encoding='utf-8').read())
    #print(type(jsonTP))
    print(jsonTP)
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'resNm', 'ForNum'))        # 컬럼에 값을 지정해 주면 해당 값만 출력
    print(type(tour_table))
    print(tour_table.head())

    print('-' * 30)
    # 'yyyymm' 이 컬럼을 색인으로 지정
    tour_table = tour_table.set_index('yyyymm')
    print(tour_table.head())

    filename = './data/immigrationTouristStat 미국(275)_(2015~2019).json'
    jsonTP = json.loads(open(filename, 'rt', encoding='utf-8').read())
    usa_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'visit_cnt'))

    # 'yyyymm' 이 컬럼을 색인으로 지정
    usa_table = usa_table.set_index('yyyymm')

    # visit_cnt 컬럼을 국가 이름으로 변경한다.
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    print(usa_table.head())

    print('-' * 30)

    filename = './data/immigrationTouristStat 영국(316)_(2015~2019).json'
    jsonTP = json.loads(open(filename, 'rt', encoding='utf-8').read())
    uk_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'visit_cnt'))

    # 'yyyymm' 이 컬럼을 색인으로 지정
    uk_table = uk_table.set_index('yyyymm')

    # visit_cnt 컬럼을 국가 이름으로 변경한다.
    uk_table = uk_table.rename(columns={'visit_cnt':'uk'})
    print(uk_table.head())

    print('-' * 30)

    filename = './data/immigrationTouristStat 일본(130)_(2015~2019).json'
    jsonTP = json.loads(open(filename, 'rt', encoding='utf-8').read())
    jp_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'visit_cnt'))

    # 'yyyymm' 이 컬럼을 색인으로 지정
    jp_table = jp_table.set_index('yyyymm')

    # visit_cnt 컬럼을 국가 이름으로 변경한다.
    jp_table = jp_table.rename(columns={'visit_cnt':'japan'})
    print(jp_table.head())

    print('-' * 30)

    filename = './data/immigrationTouristStat 중국(112)_(2015~2019).json'
    jsonTP = json.loads(open(filename, 'rt', encoding='utf-8').read())
    ch_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'visit_cnt'))

    # 'yyyymm' 이 컬럼을 색인으로 지정
    ch_table = ch_table.set_index('yyyymm')

    # visit_cnt 컬럼을 국가 이름으로 변경한다.
    ch_table = ch_table.rename(columns={'visit_cnt':'china'})
    print(ch_table.head())

    ##############
    # merge : 2개의 데이터 프레임을 합쳐주는 기능
    fv_table = pd.merge(ch_table, jp_table, left_index=True, right_index=True)
    fv_table = pd.merge(fv_table, usa_table, left_index=True, right_index=True)
    print(fv_table)

    # resNm : 방문지(목록)
    resNm = tour_table.resNm.unique()
    print(resNm)
    for tourpoint in resNm:
        setScatterGraph(tour_table, fv_table, tourpoint)

    print('-' * 30)
    print(corr_list)

if __name__ == '__main__':
    main()

