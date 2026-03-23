import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# 🔥 NEW IMPORTS (ADDED)
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_text_columns(X):
    text_cols = []
    for col in X.columns:
        if X[col].dtype == "object":
            avg_len = X[col].astype(str).str.len().mean()
            if avg_len > 5:
                text_cols.append(col)
    return text_cols


def drop_id_columns(X_train, X_test):
    id_cols = [col for col in X_train.columns if 'id' in col.lower() or 'unnamed' in col.lower()]

    print("🗑️ Dropping ID columns:", id_cols)

    X_train = X_train.drop(columns=id_cols, errors='ignore')
    X_test = X_test.drop(columns=id_cols, errors='ignore')

    return X_train, X_test


def handle_missing(X_train, X_test):
    for col in X_train.columns:
        if X_train[col].dtype in ["int64", "float64"]:
            fill = X_train[col].median()
        else:
            fill = X_train[col].mode()[0]

        X_train[col] = X_train[col].fillna(fill)
        X_test[col] = X_test[col].fillna(fill)

    return X_train, X_test


def handle_outliers(X_train, X_test):
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    return X_train, X_test


def encode_categorical(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object']).columns

    if len(cat_cols) == 0:
        print("✅ No categorical columns → skipping encoding")
        return X_train, X_test

    print("🔄 Encoding categorical columns:", list(cat_cols))

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("✅ After encoding:", list(X_train.columns))

    return X_train, X_test


def process_text(X_train, X_test, text_cols):

    train_text = X_train[text_cols].astype(str).agg(" ".join, axis=1)
    test_text = X_test[text_cols].astype(str).agg(" ".join, axis=1)

    train_text = train_text.apply(clean_text)
    test_text = test_text.apply(clean_text)

    # 🔥 UPDATED (only change: reduced max_features)
    vectorizer = TfidfVectorizer(
        max_features=300,   # 🔥 reduced from 5000 → prevents memory crash
        stop_words="english"
    )

    X_train_text = vectorizer.fit_transform(train_text).toarray()
    X_test_text = vectorizer.transform(test_text).toarray()

    return X_train_text, X_test_text, vectorizer


def preprocess(df, target_col):

    print("📊 Original columns:", list(df.columns))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("📊 Original shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("After split:", X_train.shape)

    X_train, X_test = drop_id_columns(X_train, X_test)
    print("After ID drop:", list(X_train.columns))

    text_cols = detect_text_columns(X_train)
    print("🧠 Text columns:", text_cols)

    vectorizer = None
    scaler = None

    if len(text_cols) > 0:

        X_text_train, X_text_test, vectorizer = process_text(X_train, X_test, text_cols)

        X_train = X_train.drop(columns=text_cols)
        X_test = X_test.drop(columns=text_cols)

        print("After removing text:", list(X_train.columns))

        if X_train.shape[1] > 0:

            X_train, X_test = handle_missing(X_train, X_test)
            X_train, X_test = handle_outliers(X_train, X_test)

            before_cols = list(X_train.columns)

            X_train, X_test = encode_categorical(X_train, X_test)

            if len(X_train.columns) < len(before_cols):
                print("⚠️ Columns reduced after encoding!")

            feature_columns = list(X_train.columns)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = np.hstack([X_train, X_text_train])
            X_test = np.hstack([X_test, X_text_test])

        else:
            X_train = X_text_train
            X_test = X_text_test
            feature_columns = [f"tfidf_{i}" for i in range(X_train.shape[1])]

        print("✅ Mixed/Text processed:", X_train.shape)

    else:

        print("Before numeric processing:", list(X_train.columns))

        X_train, X_test = handle_missing(X_train, X_test)
        X_train, X_test = handle_outliers(X_train, X_test)

        before_cols = list(X_train.columns)

        X_train, X_test = encode_categorical(X_train, X_test)

        if len(X_train.columns) < len(before_cols):
            print("⚠️ Columns reduced after encoding!")

        feature_columns = list(X_train.columns)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("✅ Numeric processed:", X_train.shape)

    # 🔥 NEW LOGIC (ADDED — no existing code changed)

    if X_train.shape[1] > 500:
        print("⚠️ Too many features → applying VarianceThreshold")

        selector = VarianceThreshold(threshold=0.01)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)

        print("✅ After VarianceThreshold:", X_train.shape)

    if X_train.shape[1] > 300:
        print("⚠️ Still high → applying PCA")

        pca = PCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print("✅ After PCA:", X_train.shape)

    if X_train.shape[1] < 2:
        raise ValueError("❌ Too many columns removed! Check preprocessing.")

    return X_train, X_test, y_train, y_test, {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "columns": feature_columns
    }