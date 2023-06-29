import os
import cv2
import mediapipe as mp
import pandas as pd

input_folder = "D:/Study/NCKH/2024/Discriminating dance/data/"
output_folder = "D:/Study/NCKH/2024/Discriminating dance/dataset3/"

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def extract_pose_landmarks(results):
    landmarks = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        landmarks.append(lm.x)
        landmarks.append(lm.y)
        landmarks.append(lm.z)
        landmarks.append(lm.visibility)
    return landmarks

def draw_pose_landmarks(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

# Lặp qua từng file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_video = os.path.join(input_folder, filename)
        output_csv = os.path.join(output_folder, os.path.splitext(filename)[0] + ".csv")

        # Đọc video từ đầu vào
        cap = cv2.VideoCapture(input_video)
        pose_list = []

        with mpPose.Pose() as pose:
            while True:
                ret, frame = cap.read()
                if ret:
                    # Nhận diện pose
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frameRGB)

                    if results.pose_landmarks:
                        # Ghi nhận các pose landmarks
                        pose_landmarks = extract_pose_landmarks(results)
                        pose_list.append(pose_landmarks)

                        # Vẽ khung xương lên ảnh
                        frame = draw_pose_landmarks(mpDraw, results, frame)

                    cv2.imshow("image", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    break

        # Ghi danh sách pose landmarks vào file CSV
        columns = [str(i) for i in range(len(pose_list[0]))]
        df = pd.DataFrame(pose_list, columns=columns)
        df.to_csv(output_csv, index=False)

        cap.release()
        cv2.destroyAllWindows()
