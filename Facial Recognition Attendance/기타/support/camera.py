import cv2
import os

recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
# face_detector = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

output_folder = 'data\img'  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,- 30))
    for (x, y, w, h) in faces:
        id, confidence = recogizer.predict(gray[y:y + h, x:x + w])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        img_path = os.path.join(output_folder, f'test_{len(os.listdir(output_folder)) + 1}.jpg')
        cv2.imwrite(img_path, frame)
        print(f'Captured and saved: {img_path}')

    cv2.imshow('Face Recognition', frame)

cap.release()
cv2.destroyAllWindows()
