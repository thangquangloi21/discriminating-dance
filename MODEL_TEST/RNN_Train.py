import numpy as np
import pandas as pd

from keras.layers import SimpleRNN, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
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

model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
model.save("model_rnn.h5")
# Dự đoán nhãn trên tập dữ liệu
y_pred = model.predict(X)
y_pred = (y_pred > 0.5).astype(int)

# Tính Accuracy
accuracy = accuracy_score(y, y_pred)

# Tính Precision
precision = precision_score(y, y_pred)

# Tính Recall
recall = recall_score(y, y_pred)

# Tính F1 score
f1 = f1_score(y, y_pred)

# Tính ROC-AUC
roc_auc = roc_auc_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("ROC-AUC:", roc_auc)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Tính precision, recall và thresholds cho Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y, y_pred)

# Tính false positive rate, true positive rate và thresholds cho ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y, y_pred)

# Tính diện tích dưới Precision-Recall Curve
pr_auc = auc(recall, precision)

# Tính diện tích dưới ROC Curve
roc_auc = auc(fpr, tpr)

# Vẽ Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Vẽ ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='r', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
