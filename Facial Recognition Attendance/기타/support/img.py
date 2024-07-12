import cv2
import os
import glob

def detect_and_draw_faces(input_folder, output_folder):
    
    face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    # face_detector = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

   
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))  # 형식에 맞게 수정

    for image_file in image_files:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    input_folder = 'data\ga'
    output_folder = 'data\ee'  

    detect_and_draw_faces(input_folder, output_folder)
