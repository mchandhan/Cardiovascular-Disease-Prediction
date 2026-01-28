import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.join("..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "age","gender","height","weight",
    "ap_hi","ap_lo","cholesterol",
    "gluc","smoke","alco","active"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cardio_dataset.csv")

df = pd.read_csv(CSV_PATH, sep=";")


X = df[FEATURES]
y = df["cardio"]

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)

model = LogisticRegression(max_iter=1000)
model.fit(xtrain_scaled, ytrain)

joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Model and scaler saved")
