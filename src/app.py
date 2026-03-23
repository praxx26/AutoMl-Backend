from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import boto3
import uuid
import os
import tempfile
import zipfile
import joblib

from dotenv import load_dotenv
load_dotenv()

from src.preprocessing import preprocess
from src.modelfitting import train_best_model

app = Flask(__name__)
CORS(app)

# ---------------- AWS CONFIG ----------------

BUCKET = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

print("🚀 AutoML API Running...")

# ---------------- HOME ----------------

@app.route("/")
def home():
    return "API is working 🚀"

# ---------------- UPLOAD ----------------

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files["file"]

        dataset_id = str(uuid.uuid4())

        temp_path = tempfile.mktemp(suffix=".csv")
        file.save(temp_path)

        # upload to S3
        s3.upload_file(temp_path, BUCKET, f"datasets/{dataset_id}.csv")

        df = pd.read_csv(temp_path, encoding="latin1")

        return jsonify({
            "message": "File uploaded successfully",
            "dataset_id": dataset_id,
            "columns": df.columns.tolist(),
            "preview": {
                "columns": df.columns.tolist(),
                "rows": df.head(5).to_dict(orient="records")
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- PREVIEW TARGET ----------------

@app.route("/preview-target", methods=["POST"])
def preview_target():
    try:
        data = request.json
        dataset_id = data["dataset_id"]
        target = data["target"]

        temp_path = tempfile.mktemp(suffix=".csv")

        s3.download_file(BUCKET, f"datasets/{dataset_id}.csv", temp_path)

        df = pd.read_csv(temp_path, encoding="latin1")

        return jsonify({
            "target": target,
            "unique_values": df[target].unique().tolist()[:10]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- TRAIN ----------------

@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.json
        dataset_id = data["dataset_id"]
        target = data["target"]

        temp_path = tempfile.mktemp(suffix=".csv")

        s3.download_file(BUCKET, f"datasets/{dataset_id}.csv", temp_path)

        df = pd.read_csv(temp_path, encoding="latin1")

        X_train, X_test, y_train, y_test, meta = preprocess(df, target)

        result = train_best_model(X_train, X_test, y_train, y_test)

        model = result["model"]
        label_encoder = result["label_encoder"]

        model_id = dataset_id

        # save
        model_path = f"{model_id}.pkl"
        meta_path = f"{model_id}_meta.pkl"

        joblib.dump(model, model_path)
        joblib.dump({
            "meta": meta,
            "label_encoder": label_encoder
        }, meta_path)

        # upload to S3
        s3.upload_file(model_path, BUCKET, f"models/{model_id}.pkl")
        s3.upload_file(meta_path, BUCKET, f"models/{model_id}_meta.pkl")

        return jsonify({
            "message": "Training complete",
            "model_id": model_id,
            "best_model": result["best_model_name"],
            "params": result["best_params"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        model_id = data["model_id"]
        input_data = data["input"]

        model_path = tempfile.mktemp(suffix=".pkl")
        meta_path = tempfile.mktemp(suffix=".pkl")

        s3.download_file(BUCKET, f"models/{model_id}.pkl", model_path)
        s3.download_file(BUCKET, f"models/{model_id}_meta.pkl", meta_path)

        model = joblib.load(model_path)
        meta_data = joblib.load(meta_path)

        meta = meta_data["meta"]
        label_encoder = meta_data["label_encoder"]

        df = pd.DataFrame([input_data])

        # 🔥 apply preprocessing
        X = meta["pipeline"].transform(df)

        prediction = model.predict(X)

        if label_encoder:
            prediction = label_encoder.inverse_transform(prediction)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- DOWNLOAD MODEL (ZIP) ----------------

@app.route("/download-model/<model_id>", methods=["GET"])
def download_model(model_id):
    try:
        model_path = tempfile.mktemp(suffix=".pkl")
        meta_path = tempfile.mktemp(suffix=".pkl")
        zip_path = tempfile.mktemp(suffix=".zip")

        # download from S3
        s3.download_file(BUCKET, f"models/{model_id}.pkl", model_path)
        s3.download_file(BUCKET, f"models/{model_id}_meta.pkl", meta_path)

        # create zip
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, "model.pkl")
            zipf.write(meta_path, "meta.pkl")

            zipf.writestr("README.txt", """
HOW TO USE MODEL:

import joblib

model = joblib.load("model.pkl")
meta_data = joblib.load("meta.pkl")

meta = meta_data["meta"]
label_encoder = meta_data["label_encoder"]

# preprocess input
X = meta["pipeline"].transform(input_df)

prediction = model.predict(X)

AutoML System 🚀
""")

        return send_file(
            zip_path,
            as_attachment=True,
            download_name="model_bundle.zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)