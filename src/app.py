from flask import Flask, request, jsonify,send_file, after_this_request
import zipfile
from flask_cors import CORS
import pandas as pd
import joblib
import os
import boto3
import zipfile
import uuid
import shap
import numpy as np
import tempfile
from dotenv import load_dotenv

load_dotenv()

from preprocessing import preprocess
from modelfitting import train_best_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

print(" AutoML API Running...")

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

model_cache = {}
meta_cache = {}

@app.route("/")
def home():
    return "AutoML API is working "


@app.route("/test-s3")
def test_s3():
    try:
        buckets = s3.list_buckets()
        return jsonify({
            "message": " AWS Working",
            "buckets": [b["Name"] for b in buckets["Buckets"]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    print(" Upload API hit")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print(" File:", file.filename)

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    dataset_id = str(uuid.uuid4())
    s3_key = f"datasets/{dataset_id}.csv"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            file.save(tmp.name)

            print(" Uploading to S3...")
            s3.upload_file(tmp.name, BUCKET_NAME, s3_key)
            print(" Uploaded to S3")

            df = pd.read_csv(tmp.name, encoding="latin1")

    except Exception as e:
        print(" S3 ERROR:", str(e))
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


@app.route("/preview-target", methods=["POST"])
def preview_target():
    data = request.get_json()
    target = data.get("target")
    dataset_id = data.get("dataset_id")

    if not dataset_id:
        return jsonify({"error": "dataset_id required"}), 400

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()  

        s3.download_file(BUCKET_NAME, f"datasets/{dataset_id}.csv", tmp_path)
        df = pd.read_csv(tmp_path, encoding="latin1")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if target not in df.columns:
        return jsonify({"error": "Invalid target"}), 400

    preview = df[target].head(10).tolist()

    return jsonify({"preview": preview})
@app.route("/analyze-column", methods=["POST"])
def analyze_column():
    data = request.get_json()
    column = data.get("column")
    dataset_id = data.get("dataset_id")

    if not dataset_id or not column:
        return jsonify({"error": "dataset_id and column required"}), 400

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()  

        s3.download_file(BUCKET_NAME, f"datasets/{dataset_id}.csv", tmp_path)
        df = pd.read_csv(tmp_path, encoding="latin1")

        if column not in df.columns:
            return jsonify({"error": "Invalid column"}), 400

        col_data = df[column]
        missing = int(col_data.isnull().sum())
        missing_percent = float((missing / len(df)) * 100)
        
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        outliers = 0
        stats = {}
        distribution = []

        if is_numeric:
            clean_data = col_data.dropna()
            if len(clean_data) > 0:
                stats = {
                    "min": float(clean_data.min()),
                    "max": float(clean_data.max()),
                    "mean": float(clean_data.mean()),
                    "median": float(clean_data.median())
                }
                
                # IQR Outliers
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (clean_data < (Q1 - 1.5 * IQR)) | (clean_data > (Q3 + 1.5 * IQR))
                outliers = int(outlier_mask.sum())
                
                # Distribution (10 bins)
                counts, bins = np.histogram(clean_data, bins=10)
                for i in range(len(counts)):
                    distribution.append({
                        "label": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                        "count": int(counts[i])
                    })
        else:
            clean_data = col_data.dropna().astype(str)
            if len(clean_data) > 0:
                val_counts = clean_data.value_counts()
                stats = {
                    "unique": len(val_counts),
                    "top": val_counts.index[0] if len(val_counts) > 0 else ""
                }
                # Top 10 distribution
                for val, count in val_counts.head(10).items():
                    distribution.append({
                        "label": str(val)[:15] + ("..." if len(str(val))>15 else ""),
                        "count": int(count)
                    })

        return jsonify({
            "column": column,
            "type": "numeric" if is_numeric else "categorical",
            "missing": missing,
            "missing_percent": missing_percent,
            "stats": stats,
            "outliers": outliers,
            "distribution": distribution
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    target = data.get("target")
    dataset_id = data.get("dataset_id")

    if not target or not dataset_id:
        return jsonify({"error": "target and dataset_id required"}), 400

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()  

        s3.download_file(BUCKET_NAME, f"datasets/{dataset_id}.csv", tmp_path)
        df = pd.read_csv(tmp_path, encoding="latin1")

    except Exception as e:
        return jsonify({"error": f"Dataset load failed: {str(e)}"}), 500

    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    if len(df) > 2000:
        df = df.sample(1000, random_state=42)

    X_train, X_test, y_train, y_test, meta = preprocess(df, target)

    dataset_log = {
        "step": "Dataset loaded",
        "details": [
            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            "Sample 5 rows:",
            df.head(5).to_string()
        ]
    }
    
    result = train_best_model(X_train, X_test, y_train, y_test)

    process_log = [dataset_log] + meta.get("process_log", []) + result.get("process_log", [])

    # Save a small background dataset for SHAP explainer
    try:
        bg_data = X_train.head(50) if hasattr(X_train, "head") else X_train[:50]
        if meta.get("scaler"):
            bg_data = meta["scaler"].transform(bg_data)
        elif hasattr(bg_data, "values"):
            bg_data = bg_data.values
        meta["background_data"] = bg_data
    except Exception as e:
        print("Failed to save background data:", e)

    model = result["model"]
    results = result["all_results"]
    best_model_name = result["best_model_name"]
    best_params = result["best_params"]

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

    model_id = str(uuid.uuid4())

    try:
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
            joblib.dump(model, tmp_model.name)
            s3.upload_file(tmp_model.name, BUCKET_NAME, f"models/{model_id}.pkl")

        # Save meta
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_meta:
            joblib.dump(meta, tmp_meta.name)
            s3.upload_file(tmp_meta.name, BUCKET_NAME, f"models/{model_id}_meta.pkl")

    except Exception as e:
        return jsonify({"error": f"S3 upload failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "process_log": process_log,
        "result": {
            "message": "Model trained successfully",
            "model_id": model_id,
            "best_model": best_model_name,
            "best_params": best_params,
            "leaderboard": clean_leaderboard,
            "features": meta["columns"],
            "raw_columns": meta.get("raw_columns", meta["columns"]),
            "raw_dtypes": meta.get("raw_dtypes", {}),
            "raw_mins": meta.get("raw_mins", {}),
            "raw_maxes": meta.get("raw_maxes", {}),
            "categorical_values": meta.get("categorical_values", {})
        }
    })



@app.route("/download-model/<model_id>", methods=["GET"])
def download_model(model_id):
    try:
        model_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        meta_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

        model_path = model_tmp.name
        meta_path = meta_tmp.name
        zip_path = zip_tmp.name

        model_tmp.close()
        meta_tmp.close()
        zip_tmp.close()

        s3.download_file(BUCKET_NAME, f"models/{model_id}.pkl", model_path)
        s3.download_file(BUCKET_NAME, f"models/{model_id}_meta.pkl", meta_path)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, "model.pkl")
            zipf.write(meta_path, "meta.pkl")

            zipf.writestr("README.txt", """
========================================
        AutoML Model Package 🚀
========================================

🔐 Generated by: AutoSelector AI
👨‍💻 Developer: Pragatheesh
🌐 Platform: AutoML Web App

----------------------------------------
📌 HOW TO USE MODEL
----------------------------------------

1. Install dependencies:
   pip install joblib scikit-learn pandas

2. Load model:
   import joblib
   model = joblib.load("model.pkl")
   meta_data = joblib.load("meta.pkl")

3. Preprocess input:
   meta = meta_data["meta"]
   X = meta["pipeline"].transform(input_df)

4. Predict:
   prediction = model.predict(X)

----------------------------------------
⚡ Notes:
- Ensure input_df has same columns as training data
- Handle missing values properly
----------------------------------------
🚀 Thank you for using AutoSelector AI
========================================
""")

        @after_this_request
        def cleanup(response):
            try:
                os.remove(model_path)
                os.remove(meta_path)
                os.remove(zip_path)
            except:
                pass
            return response

        return send_file(
            zip_path,
            as_attachment=True,
            download_name="model_bundle.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = data.get("input", {})
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    try:
        if model_id in model_cache:
            model = model_cache[model_id]
            meta = meta_cache[model_id]
            print(" Loaded from cache")

        else:
            print("⬇ Downloading model from S3...")

            model_file = tempfile.NamedTemporaryFile(delete=False)
            meta_file = tempfile.NamedTemporaryFile(delete=False)
            
            model_path = model_file.name
            meta_path = meta_file.name
            
            model_file.close()
            meta_file.close()

            s3.download_file(BUCKET_NAME, f"models/{model_id}.pkl", model_path)
            s3.download_file(BUCKET_NAME, f"models/{model_id}_meta.pkl", meta_path)

            model = joblib.load(model_path)
            meta = joblib.load(meta_path)

            model_cache[model_id] = model
            meta_cache[model_id] = meta

            print(" Model loaded and cached")

    except Exception as e:
        return jsonify({"error": f"Model load failed: {str(e)}"}), 500

    features = meta["columns"]
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)

    input_processed = input_df.values
    if meta.get("scaler"):
        try:
            input_processed = meta["scaler"].transform(input_df)
        except Exception as e:
            print("Scaler transform failed:", e)

    # If the model uses text features, process and append them
    if meta.get("vectorizer") and meta.get("text_cols"):
        text_cols = meta["text_cols"]
        text_data = []
        for col in text_cols:
            text_data.append(str(input_data.get(col, "")))
        text_combined = " ".join(text_data)
        text_features = meta["vectorizer"].transform([text_combined]).toarray()
        input_processed = np.hstack([input_processed, text_features])

    try:
        prediction = model.predict(input_processed)

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_processed)
            confidence = float(max(probs[0]))
            
        insights = []
        
        # Extract the underlying model if it's wrapped in a search CV object
        base_model = getattr(model, "best_estimator_", model)
        
        try:
            # Try SHAP for local explanation
            bg_data = meta.get("background_data")
            if bg_data is not None:
                # Use model.predict to support any algorithm (KNN, SVM, etc.)
                explainer = shap.Explainer(base_model.predict, bg_data)
            else:
                explainer = shap.Explainer(base_model)
                
            shap_values = explainer(input_processed)
            vals = shap_values.values[0]
            
            # Handle multi-class explanations or shap arrays
            if len(vals.shape) > 1:
                pred_class = int(prediction.tolist()[0] if hasattr(prediction, "tolist") else prediction)
                if pred_class < vals.shape[1]:
                    vals = vals[:, pred_class]
                else:
                    vals = vals[:, 0]
            
            top_idx = np.argsort(np.abs(vals))[::-1]
            for idx in top_idx:
                if idx < len(features):
                    insights.append({
                        "feature": features[idx], 
                        "importance": float(vals[idx]),
                        "type": "local"
                    })
        except Exception as shap_e:
            print("SHAP explainer failed, falling back to global:", shap_e)
            # Fallback to global feature importances
            if hasattr(base_model, "feature_importances_"):
                importances = base_model.feature_importances_
                top_idx = np.argsort(importances)[::-1]
                for idx in top_idx:
                    if idx < len(features):
                        insights.append({"feature": features[idx], "importance": float(importances[idx]), "type": "global"})
            elif hasattr(base_model, "coef_"):
                coef = base_model.coef_[0] if len(base_model.coef_.shape) > 1 else base_model.coef_
                top_idx = np.argsort(np.abs(coef))[::-1]
                for idx in top_idx:
                    if idx < len(features):
                        insights.append({"feature": features[idx], "importance": float(coef[idx]), "type": "global"})
            else:
                # If it still fails, just push a generic insight so the UI doesn't remain completely blank
                insights.append({"feature": "All features", "importance": 0.001, "type": "global"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "success",
        "result": {
            "prediction": prediction.tolist()[0] if hasattr(prediction, "tolist") else prediction,
            "confidence": float(confidence) if confidence is not None else None,
            "insights": insights
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)