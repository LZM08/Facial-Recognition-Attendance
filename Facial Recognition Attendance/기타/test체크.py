import cv2

# 데이터 가져오기
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')

# 이름-ID 매핑 딕셔너리 생성
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
for i in range(51, 61):
    names[i] = '이지명'

# 출석체크 여부
check_S1 = False
check_S2 = False
check_S3 = False
check_S4 = False
check_S5 = False
check_17 = False
all_checks_completed = False  # 모든 출석 체크 완료 여부 초기화
# 카메라 열기
cap = cv2.VideoCapture(0)
width = 1920  # 가로 해상도
height = 1080  # 세로 해상도

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 얼굴 인식용 분류기 로드
face_detector = cv2.CascadeClassifier('C:/yamigood/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
# face_detector = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

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
    highest_confidence = 0
    highest_id = 0

    for (x, y, w, h) in faces:
        # 얼굴 인식
        id, confidence = recogizer.predict(gray[y:y + h, x:x + w])

        # 얼굴 신뢰도가 제일 높은 것 저장
        if confidence > highest_confidence:
            highest_confidence = confidence
            highest_id = id

        # 얼굴 검출 부분 수정
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

    # 화면에 표시 -
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)  # 창 생성 및 크기 조절 가능 모드 설정
    cv2.imshow('Face Recognition', frame)  # 이미지 표시
    cv2.resizeWindow('Face Recognition', 800, 600)  # 창 크기 조절 (가로 800, 세로 600)

    

    if highest_confidence > 0:
        if highest_id in names:
            # print('가장 높은 신뢰도를 가진 이름:', names[highest_id], '신뢰도 평가:', highest_confidence)
            if names[highest_id] == 'S1' and highest_confidence < 120 and not check_S1:
                print(names[highest_id]+'님 출석체크 완료')
                check_S1 = True
            if names[highest_id] == 'S2' and highest_confidence < 120 and not check_S2:
                print(names[highest_id]+'님 출석체크 완료')
                check_S2 = True
            if names[highest_id] == 'S3' and highest_confidence < 120 and not check_S3:
                print(names[highest_id]+'님 출석체크 완료')
                check_S3 = True
            if names[highest_id] == 'S4' and highest_confidence < 120 and not check_S4:
                print(names[highest_id]+'님 출석체크 완료')
                check_S4 = True
            if names[highest_id] == 'S5' and highest_confidence < 120 and not check_S5:
                print(names[highest_id]+'님 출석체크 완료')
                check_S5 = True
            if names[highest_id] == '이지명' and highest_confidence < 120 and not check_17:
                print(names[highest_id]+'님 출석체크 완료')
                cv2.imwrite('checkimg/1.jpg', frame)
                check_17 = True  # 출석체크 완료 상태로 변경
               
        if check_S1 and check_S2 and check_S3 and check_S4 and check_S5 and check_17 and not all_checks_completed:
            print('모든 인원의 출석이 완료되었습니다!')
            all_checks_completed  = True

           # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
