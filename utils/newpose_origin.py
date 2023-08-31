import cv2
import numpy as np
import mediapipe as mp
import time
import speech_recognition as sr
import threading
from .fs import faceswap
import socket


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
last_state = "Unknown"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddresssPort = ('127.0.0.1', 5052)  # (로컬호스트, 포트) 사용되지 않는 포트를 이용하자!

count = 0
start_time = 0
end_time = 3


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

    with microphone as source:
        while True:
            print("STT 실행. '그만'이라고 말하면 종료")
            audio_data = recognizer.listen(source, phrase_time_limit=2)
            try:
                text = recognizer.recognize_google(
                    audio_data, language='ko-KR')
                print(f"당신이 말한 것: {text}")
                break
                if "그만" in text:
                    print("STT 종료됨.")
                    return  # 'return'을 'break'로 변경
            except sr.UnknownValueError:
                print("음성을 이해할 수 없습니다.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return  # 'return'을 'break'로 변경


def video_start():
    global last_state
    global start_time
    global count
    global end_time

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open video capture")
        exit()
    stt_thread = threading.Thread(
        target=start_stt)  # 별도의 스레드에서 STT 실행
    stt_thread.daemon = True
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while True:
            ret, frame = cap.read()
            new_state = None
            if not ret:
                print("Failed to grab frame")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if cv2.waitKey(1) == ord('a'):
                cv2.imwrite('images/cap.jpg', frame)
                faceswap('images/arrow6.jpg', 'images/cap.jpg')

            mp_drawing.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                landmarks = result.pose_landmarks.landmark

                # Get coordinates
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # 어깨와 손목의 좌표
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles
                angle_left = three_angle(shoulder_left, elbow_left, wrist_left)
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

                # Classification
                if angle_right < 90:
                    new_state = "start"
                elif is_arm_stretched_left and arm_lifted_left:
                    new_state = "ready"
                elif is_arm_stretched_right and arm_lifted_right:
                    new_state = "shot"
                elif angle_left < 90:
                    new_state = "change"
                else:
                    new_state = "Unknown"

            except Exception as e:
                print("Error occurred:", e)

            if new_state != last_state:
                print(new_state)
                last_state = new_state

                if start_time != 0:
                    end_time = time.time() - start_time
                    print(end_time)

                # if new_state == "shot":
                #     sock.sendto(str.encode(str(new_state)),
                #                 serverAddresssPort)
                if new_state != "shot" and end_time >= 3 and new_state != "Unknown":
                    start_time = 0
                    start_time = time.time()
                    sock.sendto(str.encode(str(new_state)),
                                serverAddresssPort)

                if new_state == "change":
                    try:
                        stt_thread.start()
                    except:
                        pass

            cv2.imshow('pose', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
