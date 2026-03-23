import pandas as pd
import joblib
import os

from src.detect_model_type import detect_model_type
from src.preprocessing import preprocess
from src.modelfitting import train_best_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score as ass,
    classification_report as cr,
    r2_score as r2s
)


df = pd.read_csv("data/Titanic-Dataset.csv", encoding="latin1")

target_column = "Survived"   

dtype = detect_model_type(df, target_column)


X_train, X_test, y_train, y_test, vectorizer = preprocess(df, target_column)

label_encoder = None

if dtype == "classification" and y_train.dtype == "object":
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)


model, results, scaler, fit_status = train_best_model(
    X_train, X_test, y_train, y_test
)


try:
    X_test_transformed = scaler.transform(X_test)
except:
    X_test_transformed = X_test


y_pred = model.predict(X_test_transformed)


os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

if vectorizer:
    joblib.dump(vectorizer, "models/vectorizer.pkl")

try:
    joblib.dump(X_train.columns, "models/features.pkl")
except:
    pass

if label_encoder:
    joblib.dump(label_encoder, "models/label_encoder.pkl")

print("✅ Training completed and models saved")


print("\n📊 Model Performance:\n")

if dtype == 'classification':
    print("Accuracy :", ass(y_test, y_pred))
    print("\nClassification Report:\n")
    print(cr(y_test, y_pred))

else:
    print("R2 Score :", r2s(y_test, y_pred))