import os
import cv2

input_path = 'E:\얼굴인식 출석체크\data\jm'
output_path = 'E:\얼굴인식 출석체크\data\jm_resized'

# 원하는 이미지 크기 설정
target_image_size = (100, 100)

# 출력 폴더 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 이미지 파일들 가져오기
image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jpg')]

for image_path in image_paths:
    # 이미지 불러오기
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # 이미지 크기 조정
    resized_img = cv2.resize(img, target_image_size)
    
    # 새로운 파일 경로 생성
    new_image_path = os.path.join(output_path, os.path.basename(image_path))
    
    # 이미지 저장
    cv2.imwrite(new_image_path, resized_img)

print("이미지 크기 조정 완료")
