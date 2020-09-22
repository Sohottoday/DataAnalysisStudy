
# os 모듈 : Operating System(운영체제)와 관련된 모듈
# 폴더/파일 생성, 수정, 삭제 등등
# 파일과 관련된 여러 속성 정보
import os

# 폴더 구분자를 사용할 때 슬래시는 한번, 역슬래시는 두 개를 적어야 한다.
myfolder = 'd:\\'
newpath = os.path.join(myfolder, 'sample')

try:
    os.mkdir(path=newpath)

    for idx in range(1, 11):
        newfile = os.path.join(newpath, 'folder'+str(idx).zfill(2))     # zfill 은 숫자를 채울 때 숫자를 지정해준 만큼 앞자리에 0 추가
        os.mkdir(path=newfile)


except FileExistsError as err:
    print('폴더가 존재합니다. 우선 삭제해 주세요')


print('finished')