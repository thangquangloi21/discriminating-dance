import os

import numpy
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential,load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
no_of_timesteps = 10
X = []
y = []
# đọc dữ liệu
inp_dance1 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/heyhey"
inp_dance2 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/Kyngucfan"
inp_dance3 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/THUYENQUEN"
inp_dance4 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/tronghoa"
inp_dance5 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/Vietnam"

# dance_1
for filename in os.listdir(inp_dance1):
    if filename.endswith(".csv"):
        file_csv = os.path.join(inp_dance1, filename)
        dance1_df = pd.read_csv(file_csv)
        dataset = dance1_df.iloc[:, 0:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(0)
        print("Dance1: ", file_csv)

# dance_2
for filename in os.listdir(inp_dance2):
    if filename.endswith(".csv"):
        file_csv = os.path.join(inp_dance2, filename)
        dance2_df = pd.read_csv(file_csv)
        dataset = dance2_df.iloc[:, 0:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(1)
        print("Dance2: ", file_csv)

# dance_3
for filename in os.listdir(inp_dance3):
    if filename.endswith(".csv"):
        file_csv = os.path.join(inp_dance3, filename)
        dance3_df = pd.read_csv(file_csv)
        dataset = dance3_df.iloc[:, 0:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(2)
        print("Dance3: ", file_csv)

# dance_4
for filename in os.listdir(inp_dance4):
    if filename.endswith(".csv"):
        file_csv = os.path.join(inp_dance4, filename)
        dance4_df = pd.read_csv(file_csv)
        dataset = dance4_df.iloc[:, 0:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(3)
        print("Dance4: ", file_csv)

# dance_5
for filename in os.listdir(inp_dance5):
    if filename.endswith(".csv"):
        file_csv = os.path.join(inp_dance5, filename)
        dance5_df = pd.read_csv(file_csv)
        dataset = dance5_df.iloc[:, 0:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(4)
        print("Dance5: ", file_csv)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, x_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= 0.2)

## Xây dựng mô hình
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

## Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16 , batch_size=32, validation_data=(x_val, y_val))
model.save("model_multi.h5")

model = load_model("model_multi.h5")
loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss)
print("acc:", acc)
# Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred, average='macro')

print("F1 score:", f1)
