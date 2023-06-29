import os
import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import threading

input_folder = "D:/Study/NCKH/2024/Discriminating dance/data/"
output_folder = "D:/Study/NCKH/2024/Discriminating dance/dataset3/"
print(tf.__version__)
# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils

# Bật GPU acceleration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Chỉ cấu hình sử dụng GPU đầu tiên
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU acceleration enabled.")
    except RuntimeError as e:
        print(e)

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

        # Xử lý mỗi video trong một luồng riêng biệt
        def process_video(input_video, output_csv):
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

        # Tạo một luồng mới để xử lý video
        thread = threading.Thread(target=process_video, args=(input_video, output_csv))
        thread.start()

        # Chờ cho đến khi luồng xử lý video hoàn thành
        thread.join()

print("Processing complete.")
