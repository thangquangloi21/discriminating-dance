import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def read_data(inp_dance, label):
    for filename in os.listdir(inp_dance):
        if filename.endswith(".csv"):
            file_csv = os.path.join(inp_dance, filename)
            dance_df = pd.read_csv(file_csv)
            dataset = dance_df.iloc[:, 0:].values
            n_sample = len(dataset)
            for i in range(no_of_timesteps, n_sample):
                X.append(dataset[i - no_of_timesteps:i, :])
                y.append(label)
            print(file_csv)


no_of_timesteps = 10
X = []
y = []
# đọc dữ liệu
inp_dance1 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/heyhey"
inp_dance2 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/Kyngucfan"
inp_dance3 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/THUYENQUEN"
inp_dance4 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/tronghoa"
inp_dance5 = "/home/loiii/Desktop/Study/NCKH/discriminating-dance/DATA/Vietnam"


read_data(inp_dance1, 0)
read_data(inp_dance2, 1)
read_data(inp_dance3, 2)
read_data(inp_dance4, 3)
read_data(inp_dance5, 4)
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, x_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

# Xây dựng mô hình
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

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(x_val, y_val))
model.save("model_lstm.h5")

model = load_model("model_lstm.h5")
loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss)
print("acc:", acc)

# Đánh giá mô hình f1 scope
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred, average='macro')

print("F1 score:", f1)
