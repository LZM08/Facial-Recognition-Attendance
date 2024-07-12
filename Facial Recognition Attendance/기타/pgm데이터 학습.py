# import os
# import cv2
# import sys
# from PIL import Image
# import numpy as np

# if not os.path.exists('trainer'):
#     os.makedirs('trainer')

# def getImageAndLabels(path):
#     facesSamples = []
#     ids = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
#     face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    
#     for imagePath in imagePaths:
#         PIL_img = Image.open(imagePath).convert('L')  # 이미지를 그레이스케일로 변환
#         img_numpy = np.array(PIL_img, 'uint8')
#         faces = face_detector.detectMultiScale(img_numpy)
#         name = os.path.split(imagePath)[1].split('_')[0]
#         for x, y, w, h in faces:
#             facesSamples.append(img_numpy[y:y+h, x:x+w])
#             ids.append(name)

#     return facesSamples, ids

# if __name__ == '__main__':
#     path = 'data/jm'
#     faces, ids = getImageAndLabels(path)
    
#     # 'ids' 리스트를 'int32' 데이터 타입의 numpy 배열로 변환
#     ids = np.array(ids, dtype=np.int32)
    
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, ids)
#     recognizer.write('trainer/trainer.yml')





import os
import cv2
from PIL import Image
import numpy as np

# 이미지를 PGM 형식으로 변환하여 저장하는 함수
def convert_images_to_pgm(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 입력 폴더의 이미지 파일들을 가져옴
    image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path)] 
    
    # 각 이미지를 PGM 형식으로 변환하여 저장
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # 흑백 이미지로 변환
        new_image_path = os.path.join(output_path, os.path.basename(image_path).split('.')[0] + '.pgm')
        img.save(new_image_path)

# 이미지와 레이블을 가져오는 함수
def getImageAndLabels(path):
    facesSamples = []
    ids = []
    id_map = {}
    current_id = 0
    
    # PGM 형식의 이미지 파일들을 가져옴
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
    face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        
        name = os.path.splitext(os.path.split(imagePath)[1])[0].split('_')[0] + '_' + os.path.splitext(os.path.split(imagePath)[1])[0].split('_')[1]

        if name not in id_map:
            id_map[name] = current_id
            current_id += 1
        
        for x, y, w, h in faces:
            facesSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id_map[name])

    return facesSamples, ids

if __name__ == '__main__':
    input_path = 'E:\얼굴인식 출석체크\data\jm'  # 입력 이미지 폴더 경로
    output_path = 'E:\얼굴인식 출석체크\data\jm_pgm'  # PGM 형식으로 변환한 이미지 저장 폴더 경로
    
    # 이미지를 PGM 형식으로 변환하여 저장
    convert_images_to_pgm(input_path, output_path)
    
    # PGM 이미지와 레이블을 가져와서 학습
    faces, ids = getImageAndLabels(output_path)
    ids = np.array(ids, dtype=np.int32)
    
    # 얼굴 인식 모델 생성 및 학습
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    recognizer.write('trainer/trainer.yml')  # 학습된 모델 저장


# import os
# import cv2
# import sys
# from PIL import Image
# import numpy as np

# if not os.path.exists('trainer'):
#     os.makedirs('trainer')

# def getImageAndLabels(path):
#     facesSamples = []
#     ids = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
#     face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    
#     for imagePath in imagePaths:
#         PIL_img = Image.open(imagePath).convert('L')
#         img_numpy = np.array(PIL_img, 'uint8')
#         faces = face_detector.detectMultiScale(img_numpy)
#         name = os.path.split(imagePath)[1].split('_')[0]
#         for x, y, w, h in faces:
#             facesSamples.append(img_numpy[y:y+h, x:x+w])
#             ids.append(name)

#     return facesSamples, ids

# if __name__ == '__main__':
#     path = 'data/jm'
#     faces, ids = getImageAndLabels(path)
    
#     # 'ids' 리스트를 'int32' 데이터 타입의 numpy 배열로 변환
#     ids = np.array(ids, dtype=np.int32)
    
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, ids)
#     recognizer.write('trainer/trainer.yml')
