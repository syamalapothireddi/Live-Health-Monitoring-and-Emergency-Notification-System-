import pandas as pd
import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ================= LOAD DATASET =================
df = pd.read_csv("health_dataset_3class.csv")

# ================= FEATURES =================
X = df[['HeartRate','SpO2','TemperatureF','FallDetected']]
y = df['Condition']

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODELS =================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

results = {}

# ================= TRAIN & EVALUATE =================
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "y_pred": y_pred
    }

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")

# ================= BEST MODEL =================
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print("\n🔥 BEST MODEL:", best_model_name)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================= ACCURACY GRAPH =================
names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in names]

plt.figure()
plt.bar(names, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()

# ================= SERIAL CONNECTION =================
ser = serial.Serial('COM9', 115200)  # 🔴 Change if needed
time.sleep(2)

print("\n📡 Reading Serial Data...\n")

# ================= REAL-TIME PREDICTION =================
while True:
    try:
        # ✅ FIX 1: UTF-8 error removed
        line = ser.readline().decode(errors='ignore').strip()

        if line:
            print("Raw:", line)

            data = line.split(',')

            if len(data) == 4:
                hr = float(data[0])
                spo2 = float(data[1])
                temp = float(data[2])
                fall = float(data[3])

                # ✅ FIX 2: Use DataFrame with column names (NO WARNING)
                input_df = pd.DataFrame([[hr, spo2, temp, fall]],
                                        columns=['HeartRate','SpO2','TemperatureF','FallDetected'])

                input_scaled = scaler.transform(input_df)

                prediction = best_model.predict(input_scaled)

                print(f"Prediction: {prediction[0]}\n")

    except Exception as e:
        print("Error:", e)
