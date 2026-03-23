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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

print("🚀 AutoML API Running...")

# ---------------- AWS CONFIG ----------------
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

print("AWS KEY:", AWS_ACCESS_KEY)
print("AWS REGION:", AWS_REGION)
print("BUCKET:", BUCKET_NAME)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# 🔥 Model cache (NEW)
model_cache = {}
meta_cache = {}

# ---------------- HOME ----------------
@app.route("/")
def home():
    return "AutoML API is working 🚀"


# ---------------- TEST S3 ----------------
@app.route("/test-s3")
def test_s3():
    try:
        buckets = s3.list_buckets()
        return jsonify({
            "message": "✅ AWS Working",
            "buckets": [b["Name"] for b in buckets["Buckets"]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    print("🔥 Upload API hit")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print("📁 File:", file.filename)

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    dataset_id = str(uuid.uuid4())
    s3_key = f"datasets/{dataset_id}.csv"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            file.save(tmp.name)

            print("⬆️ Uploading to S3...")
            s3.upload_file(tmp.name, BUCKET_NAME, s3_key)
            print("✅ Uploaded to S3")

            df = pd.read_csv(tmp.name, encoding="latin1")

    except Exception as e:
        print("❌ S3 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    preview = {
        "columns": list(df.columns),
        "rows": df.head(5).to_dict(orient="records")
    }

    return jsonify({
        "message": "File uploaded successfully",
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "preview": preview
    })


# ---------------- TARGET PREVIEW ----------------
@app.route("/preview-target", methods=["POST"])
def preview_target():
    data = request.get_json()
    target = data.get("target")
    dataset_id = data.get("dataset_id")

    if not dataset_id:
        return jsonify({"error": "dataset_id required"}), 400

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        s3.download_file(BUCKET_NAME, f"datasets/{dataset_id}.csv", tmp_file.name)
        df = pd.read_csv(tmp_file.name, encoding="latin1")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if target not in df.columns:
        return jsonify({"error": "Invalid target"}), 400

    preview = df[target].head(10).tolist()

    return jsonify({
        "preview": preview
    })


# ---------------- TRAIN (DISABLED) ----------------
@app.route("/train", methods=["POST"])
def train_model():
    return jsonify({
        "message": "❌ Training disabled on server. Train locally and upload model to S3."
    }), 400


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = data.get("input", {})
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    try:
        # 🔥 Load from cache if already loaded
        if model_id in model_cache:
            model = model_cache[model_id]
            meta = meta_cache[model_id]
            print("⚡ Loaded from cache")

        else:
            print("⬇️ Downloading model from S3...")

            model_file = tempfile.NamedTemporaryFile(delete=False)
            meta_file = tempfile.NamedTemporaryFile(delete=False)

            s3.download_file(BUCKET_NAME, f"models/{model_id}.pkl", model_file.name)
            s3.download_file(BUCKET_NAME, f"models/{model_id}_meta.pkl", meta_file.name)

            model = joblib.load(model_file.name)
            meta = joblib.load(meta_file.name)

            # 🔥 Store in cache
            model_cache[model_id] = model
            meta_cache[model_id] = meta

            print("✅ Model loaded and cached")

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