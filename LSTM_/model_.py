import numpy as np
import pandas as pd
from keras.layers import GRU, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
dance1_df = pd.read_csv("D:/Study/NCKH/2024/Discriminating dance/dataset/lalala.csv")
dance2_df = pd.read_csv("D:/Study/NCKH/2024/Discriminating dance/dataset/quynhalee.csv")
dance3_df = pd.read_csv("D:/Study/NCKH/2024/Discriminating dance/dataset/vevoiem.csv")

X = []
y = []
no_of_timesteps = 10

# Dance 1
dataset = dance1_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(0)

# Dance 2
dataset = dance2_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(1)

# Dance 3
dataset = dance3_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(2)



X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
model.save("model_multi.h5")
