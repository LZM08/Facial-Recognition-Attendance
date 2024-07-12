import cv2


recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('E:/얼굴인식 출석체크/trainer/trainer.yml')

names = {
    0: 'Unknown'
}
for i in range(1, 11):
    names[i] = 'S1'
for i in range(11, 21):
    names[i] = 'S2'
for i in range(21, 31):
    names[i] = 'S3'
for i in range(31, 41):
    names[i] = 'S4'
for i in range(41, 51):
    names[i] = 'S5'


img = cv2.imread('test/2_(10).pgm')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
faces = face_detector.detectMultiScale(gray)
highest_confidence = float('inf')
highest_confidence_id = -1

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    id, confidence = recogizer.predict(gray[y:y + h, x:x + w])
    print('이름:', names[id], '신뢰도 평가:', confidence)
    
    if confidence < highest_confidence:
        highest_confidence = confidence
        highest_confidence_id = id

print('신뢰도가 가장 높은 이름:', names[highest_confidence_id], '신뢰도 평가:', highest_confidence)



cv2.imshow('result',img)
cv2.waitKey(0)  