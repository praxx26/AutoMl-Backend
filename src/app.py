from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import boto3
import uuid
import tempfile
from dotenv import load_dotenv


load_dotenv()

from src.preprocessing import preprocess
from src.modelfitting import train_best_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},supports_credentials=True)

print("🚀 AutoML API Running...")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


@app.route("/")
def home():
    return "AutoML API is working 🚀"


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    os.makedirs("data", exist_ok=True)

    filepath = os.path.join("data", "uploaded.csv")
    file.save(filepath)

    df = pd.read_csv(filepath, encoding="latin1")

    preview = {
        "columns": list(df.columns),
        "rows": df.head(5).to_dict(orient="records")
    }

    return jsonify({
        "message": "File uploaded successfully",
        "columns": list(df.columns),
        "preview": preview
    })


# ---------------- TARGET PREVIEW ----------------
@app.route("/preview-target", methods=["POST"])
def preview_target():
    data = request.get_json()
    target = data.get("target")

    filepath = "data/uploaded.csv"

    if not os.path.exists(filepath):
        return jsonify({"error": "No dataset uploaded"}), 400

    df = pd.read_csv(filepath, encoding="latin1")

    if target not in df.columns:
        return jsonify({"error": "Invalid target"}), 400

    preview = df[target].head(10).tolist()

    return jsonify({
        "preview": preview
    })


# ---------------- TRAIN ----------------
@app.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    target = data.get("target")

    if not target:
        return jsonify({"error": "Target column required"}), 400

    filepath = "data/uploaded.csv"

    if not os.path.exists(filepath):
        return jsonify({"error": "No dataset uploaded"}), 400

    df = pd.read_csv(filepath, encoding="latin1")

    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    # Preprocess
    X_train, X_test, y_train, y_test, meta = preprocess(df, target)

    # Train
    result = train_best_model(X_train, X_test, y_train, y_test)

    model = result["model"]
    results = result["all_results"]
    best_model_name = result["best_model_name"]
    best_params = result["best_params"]

    # Leaderboard
    leaderboard = sorted(results, key=lambda x: x["test_score"], reverse=True)

    clean_leaderboard = []
    for r in leaderboard:
        clean_leaderboard.append({
            "model": r["name"],
            "train_score": float(r["train_score"]),
            "test_score": float(r["test_score"]),
            "cv_score": float(r["cv_score"]),
            "params": r["params"],
            "fit_status": r["fit_status"]
        })

    # 🔥 SAVE MODEL TO S3
    model_id = str(uuid.uuid4())

    # Save model temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
        joblib.dump(model, tmp_model.name)
        model_path = tmp_model.name

    s3.upload_file(model_path, BUCKET_NAME, f"{model_id}.pkl")

    # Save meta temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_meta:
        joblib.dump(meta, tmp_meta.name)
        meta_path = tmp_meta.name

    s3.upload_file(meta_path, BUCKET_NAME, f"{model_id}_meta.pkl")

    return jsonify({
        "message": "Model trained successfully",
        "model_id": model_id,   # 🔥 IMPORTANT
        "best_model": best_model_name,
        "best_params": best_params,
        "leaderboard": clean_leaderboard,
        "features": meta["columns"]
    })


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = data.get("input", {})
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    try:
        # Download model from S3
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        meta_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")

        s3.download_file(BUCKET_NAME, f"{model_id}.pkl", model_file.name)
        s3.download_file(BUCKET_NAME, f"{model_id}_meta.pkl", meta_file.name)

        model = joblib.load(model_file.name)
        meta = joblib.load(meta_file.name)

    except Exception as e:
        return jsonify({"error": f"Model load failed: {str(e)}"}), 500

    features = meta["columns"]

    input_df = pd.DataFrame([input_data])

    for col in features:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[features]

    try:
        prediction = model.predict(input_df)

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)
            confidence = float(max(probs[0]))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": str(prediction[0]),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)