import cv2
import mediapipe as mp
import numpy as np
from yolov5.models import *

import torch
# 모델 이름, 버전, 저장 경로 설정
model = torch.hub.load("ultralytics/yolov5", "yolov5-6.0", path="yolov5.pt")

# 모델 추론 수행
results = model(image)

# yolov5 모델 로드
model = yolov5("yolov5.pt")

# MediaPipe 손 추적 모델 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    # 웹캠으로부터 프레임 읽어오기
    ret, frame = cap.read()

    # yolov5 모델로 추론 수행
    results = model(frame)

    # 손 감지 결과 확인
    for detection in results.pred[0]:
        if int(detection[5]) == 0:  # 손 클래스 인덱스 확인
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

            # 손 영역 이미지 추출
            hand_image = frame[y1:y2, x1:x2]

            # MediaPipe 모델로 손 추적 수행
            results = hands.process(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))

            # 손 관절 좌표 추출
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        print(landmark.x, landmark.y, landmark.z)

    # 프레임 출력
    cv2.imshow("Webcam", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
