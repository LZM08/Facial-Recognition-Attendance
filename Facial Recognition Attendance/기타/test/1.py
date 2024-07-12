import os

folder_path = 'E:\얼굴인식 출석체크\data\jm_pgm'  # 해당 폴더의 경로로 수정하세요

# 폴더 안의 파일 리스트를 가져옴
file_list = os.listdir(folder_path)

# 파일 개수 출력
file_count = len(file_list)
print(f'폴더 내 파일 개수: {file_count}개')
