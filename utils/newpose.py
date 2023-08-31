import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import speech_recognition as sr
import threading
import socket

from fs import faceswap


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
last_state = "Unknown"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddresssPort = ('192.168.1.18', 5052)  # (로컬호스트, 포트) 사용되지 않는 포트를 이용하자!

count = 0
start_time = 0
end_time = 0
idx = 2
condition = False


def three_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# TODO STT를 위한 함수
def start_stt():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        with microphone as source:
            print("STT 실행. '그만'이라고 말하면 종료")
            audio_data = recognizer.listen(source, phrase_time_limit=2)
            try:
                text = recognizer.recognize_google(
                    audio_data, language='ko-KR')
                print(f"당신이 말한 것: {text}")
                if "그만" in text:
                    print("STT 종료됨.")
                    break  # 'return'을 'break'로 변경
            except sr.UnknownValueError:
                print("음성을 이해할 수 없습니다.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return  # 'return'을 'break'로 변경


def video_start():
    global last_state
    global start_time
    global count
    global idx
    global condition

    stt_thread = threading.Thread(
        target=start_stt)  # 별도의 스레드에서 STT 실행
    stt_thread.daemon = True

    cap = cv2.VideoCapture(1)

    # knn
    df = pd.read_csv(
        'D:\\Users\\user\\projects\\mn\\utils\\hand2.csv', header=None)
    x = df.iloc[:, :-1].to_numpy().astype(np.float32)
    y = df.iloc[:, -1].to_numpy().astype(np.float32)

    # cv2의 knn 머신러닝 알고리즘을 사용할 수 있다.
    knn = cv2.ml.KNearest_create()
    knn.train(x, cv2.ml.ROW_SAMPLE, y)

    # tcp 스레드
    tcp_thread = threading.Thread(
        target=faceswap, args=(['images/cap.jpg']))
    tcp_thread.daemon = True

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_state_change_time = 0
        while cap.isOpened():
            _, frame = cap.read()

            if cv2.waitKey(1) == ord('a') and not tcp_thread.is_alive():
                cv2.imwrite('images/cap.jpg', frame)
                tcp_thread.start()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Right hand
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if results.right_hand_landmarks is not None:

                joint = np.zeros((21, 3))
                for j, landmark in enumerate(results.right_hand_landmarks.landmark):

                    joint[j] = [landmark.x, landmark.y, landmark.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,
                            10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1

                v = v / (np.linalg.norm(v, axis=1))[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[
                    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                angle = np.degrees(angle)
                data = np.array([angle], dtype=np.float32)

                ret, rdata, neig, dist = knn.findNearest(
                    data, 5)  # 가장 가까운 5개를 기준
                idx = int(rdata[0][0])

            # Pose Detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    shoulder_left = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_left = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                    elbow_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_left = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
                    wrist_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
                    # 어깨와 손목의 좌표
                    shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                    wrist_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angles
                    angle_left = three_angle(
                        shoulder_left, elbow_left, wrist_left)
                    angle_right = three_angle(
                        shoulder_right, elbow_right, wrist_right)

                    # 어깨와 손목의 y좌표가 비슷한지 확인 (옆으로 팔이 뻗어 있는지)
                    y_threshold = 0.05
                    x_threshold = 0.05
                    is_arm_stretched_left = abs(shoulder_left[1] - wrist_left[1]) < y_threshold and abs(
                        shoulder_left[0] - wrist_left[0]) > x_threshold
                    is_arm_stretched_right = abs(shoulder_right[1] - wrist_right[1]) < y_threshold and abs(
                        shoulder_right[0] - wrist_right[0]) > x_threshold

                    # 팔이 어느 정도 위로 올라갔는지 확인
                    arm_lifted_left = elbow_left[1] > shoulder_left[1]
                    arm_lifted_right = elbow_right[1] > shoulder_right[1]

                except Exception as e:
                    print("Error occurred:", e)

            '''
            액션 트리거

            스타트 조건
                왼손을 가슴에 대면서 오른손 검지를 펴기. 오른팔은 길게 뻗어야 함. 

            레디 조건
                왼쪽 팔을 길게 뻗어야 하고, 오른손 주먹을 쥐면서 왼쪽팔과 반대편,
                오른쪽 팔꿈치를 어깨보다 높이 올려야 함. 우상향 곡선을 그리며 올라가야 함.
                
            샷 조건
                왼팔과 손바닥을 활짝 펴야 함. 
                
            체인지 조건
                왼팔의 각도 70도에서 90도 사이이면서 팔꿈치가 어깨와 비슷하거나 높게, 손바닥은 활짝 펴기
        
            '''

            # idx 0 :  주먹, idx 0 : 보, idx 3 : 손가락 가르키기
            # if angle_left <= 20 and idx == 3 and is_arm_stretched_right:
            #     new_state = "start"
            # elif is_arm_stretched_left and idx == 0 and wrist_right[0] <= 0.5 and arm_lifted_right == False:
            #     new_state = "ready"
            # elif is_arm_stretched_left and idx == 1 and wrist_right[0] <= 0.5:
            #     new_state = "shot"
            # elif arm_lifted_left == False and angle_left <= 30:
            #     new_state = "change"
            # else:
            #     new_state = "Unknown"
            # elif is_arm_stretched_left and is_arm_stretched_right and idx == 1 and wrist_right[0] <= 0.4:

            if angle_left <= 20 and idx == 3 and is_arm_stretched_right:
                new_state = "start"
            elif is_arm_stretched_left and idx == 0 and wrist_right[0] <= 0.4 and arm_lifted_right == False:
                new_state = "ready"
            elif is_arm_stretched_left and idx == 1 and wrist_right[0] <= 0.4:
                new_state = "shot"
            elif arm_lifted_left == False and angle_left <= 40:
                new_state = "change"
            else:
                new_state = "Unknown"

            '''
                소켓 전송 조건
                
                이전 동작와 현재 동작이 같으면 안됨
                샷이면 시간 딜레이 없이 바로 쏘기 가능
                나머지는 3초를 기다려야 함
                언노운이면 통과 못함
            '''

            if new_state != last_state and new_state != "Unknown":
                last_state = new_state
                count += 1

                if new_state == "shot":
                    print(new_state, count)
                    sock.sendto(str.encode(str(new_state)),
                                serverAddresssPort)

                if new_state != "Unknown" and new_state != "shot":
                    print(new_state, count)
                    start_time = 0

                    sock.sendto(str.encode(str(new_state)),
                                serverAddresssPort)

                if new_state == "change":
                    try:
                        stt_thread.start()
                    except:
                        pass

            image = cv2.resize(image, (1280, 720))
            cv2.imshow('vid', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    video_start()
