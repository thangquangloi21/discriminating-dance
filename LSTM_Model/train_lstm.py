import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
lalala_df = pd.read_csv("D:/Study/NCKH/2024/Discriminating dance/dataset/lalala.txt")
vevoiem_df = pd.read_csv("D:/Study/NCKH/2024/Discriminating dance/dataset/vevoiem.txt")
X = []
y = []
no_of_timesteps = 10

dataset = lalala_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(1)

dataset = vevoiem_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model_LSTM.h5")
# Dự đoán nhãn trên tập test
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Tính F1 score
f1 = f1_score(y_test, y_pred)

print("F1 score:", f1)
