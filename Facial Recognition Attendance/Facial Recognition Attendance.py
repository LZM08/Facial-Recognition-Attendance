import cv2

print("OpenCV 버전:", cv2.__version__)
# 데이터 가져오기
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('Data Learning/dataLearning.yml')

# 이름-ID 매핑 딕셔너리 생성
names = {
    
}
for i in range(0, 30):
    names[i] = '이지명'
for i in range(30, 60):
    names[i] = '배상준'

# 출석체크 
check_617 = False
check_712 = False
all_checks_completed = False  
# 카메라 열기
cap = cv2.VideoCapture(0)
width = 1280  
height = 1018  

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


cap.set(cv2.CAP_PROP_FPS, 30)
# 얼굴 인식용 분류기 로드
# face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

while True:
    # 카메라로부터 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 최고 신뢰도 초기화
    confidence = 0
    id = 0

    for (x, y, w, h) in faces:
        # 얼굴 인식
        id, confidence = recogizer.predict(gray[y:y + h, x:x + w])

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL) 
    cv2.imshow('Face Recognition', frame)  
    

    

    if confidence > 0:
        if id in names:
            print('가장 높은 신뢰도를 가진 이름:',names[id], '신뢰도 평가:', confidence)
            if names[id] == '이지명' and confidence < 70 and not check_617:
                print(names[id] +'님 출석체크 완료')
                cv2.imwrite('checkimg/617.jpg', frame)
                check_617 = True
            if names[id] == '배상준' and confidence < 70 and not check_712:
                print(names[id] +'님 출석체크 완료')
                cv2.imwrite('checkimg/712.jpg', frame)
                check_712 = True


        if check_712 and check_617 and not all_checks_completed:
            print('모든 인원의 출석이 완료되었습니다!')
            all_checks_completed  = True

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
