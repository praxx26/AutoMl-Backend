import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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

    print(" Dropping ID columns:", id_cols)

    X_train = X_train.drop(columns=id_cols, errors='ignore')
    X_test = X_test.drop(columns=id_cols, errors='ignore')

    return X_train, X_test, id_cols


def handle_missing(X_train, X_test):
    missing_before = int(X_train.isnull().sum().sum() + X_test.isnull().sum().sum())
    
    for col in X_train.columns:

        if X_train[col].dtype in ["int64", "float64"]:
            skewness = X_train[col].skew()
            if(skewness < 0.5):
                fill=X_train[col].mean()
            else:
                fill = X_train[col].median()
        else:
            if not X_train[col].mode().empty:
                fill = X_train[col].mode()[0]
            else:
                fill = "Unknown"

        X_train[col] = X_train[col].fillna(fill)
        X_test[col] = X_test[col].fillna(fill)
        
    missing_after = int(X_train.isnull().sum().sum() + X_test.isnull().sum().sum())

    log = {
        "step": "Missing values handled",
        "details": [
            f"Missing values before imputation: {missing_before}",
            f"Missing values after imputation: {missing_after}"
        ]
    }
    return X_train, X_test, log


def handle_outliers(X_train, X_test):
    numeric_cols = X_train.select_dtypes(
        include=['int64', 'float64']
    ).columns

    outliers_handled = 0
    
    for col in numeric_cols:
        skewness = X_train[col].skew()

        if abs(skewness) > 1:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        else:
            mean = X_train[col].mean()
            std = X_train[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std
            
        outliers_count = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
        outliers_handled += outliers_count
        
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    log = {
        "step": "Outliers handled",
        "details": [
            f"Numeric columns checked: {len(numeric_cols)}",
            f"Total outliers clipped: {outliers_handled}"
        ]
    }
    return X_train, X_test, log


def encode_categorical(X_train, X_test):
    cat_cols = list(X_train.select_dtypes(include=['object']).columns)

    if len(cat_cols) == 0:
        print(" No categorical columns -> skipping encoding")
        return X_train, X_test, None

    print("Encoding categorical columns:", cat_cols)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(" After encoding:", list(X_train.columns))

    log = {
        "step": "Features encoded",
        "details": [
            f"Categorical columns encoded: {cat_cols}",
            f"Total features after encoding: {X_train.shape[1]}"
        ]
    }
    return X_train, X_test, log


def process_text(X_train, X_test, text_cols):

    train_text = X_train[text_cols].astype(str).agg(" ".join, axis=1)
    test_text = X_test[text_cols].astype(str).agg(" ".join, axis=1)

    train_text = train_text.apply(clean_text)
    test_text = test_text.apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=300, 
        stop_words="english"
    )

    X_train_text = vectorizer.fit_transform(train_text).toarray()
    X_test_text = vectorizer.transform(test_text).toarray()

    return X_train_text, X_test_text, vectorizer


def preprocess(df, target_col):
    process_log = []

    print(" Original columns:", list(df.columns))

    # Drop rows where the target is missing
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    process_log.append({
        "step": "Target column separated",
        "details": [
            f"Target column: {target_col}",
            f"Feature columns count: {X.shape[1]}"
        ]
    })

    print(" Original shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    process_log.append({
        "step": "Train-test split performed",
        "details": [
            f"Train set size: {X_train.shape[0]} rows",
            f"Test set size: {X_test.shape[0]} rows",
            f"Split ratio: 80% Train / 20% Test"
        ]
    })

    print("After split:", X_train.shape)

    X_train, X_test, dropped_id_cols = drop_id_columns(X_train, X_test)
    if dropped_id_cols:
        process_log.append({
            "step": "ID columns dropped",
            "details": [f"Dropped columns: {dropped_id_cols}"]
        })

    raw_columns = list(X_train.columns)
    raw_dtypes = {col: str(X_train[col].dtype) for col in raw_columns}
    
    raw_mins = {}
    raw_maxes = {}
    for col in raw_columns:
        if np.issubdtype(X_train[col].dtype, np.number):
            raw_mins[col] = float(X_train[col].min())
            raw_maxes[col] = float(X_train[col].max())
    
    categorical_values = {}
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            categorical_values[col] = list(X_train[col].dropna().unique())

    text_cols = detect_text_columns(X_train)
    print(" Text columns:", text_cols)

    vectorizer = None
    scaler = None

    if len(text_cols) > 0:
        process_log.append({
            "step": "Text columns detected",
            "details": [f"Text columns processed with TF-IDF: {text_cols}"]
        })
        X_text_train, X_text_test, vectorizer = process_text(X_train, X_test, text_cols)

        X_train = X_train.drop(columns=text_cols)
        X_test = X_test.drop(columns=text_cols)

        if X_train.shape[1] > 0:
            X_train, X_test, missing_log = handle_missing(X_train, X_test)
            process_log.append(missing_log)
            
            X_train, X_test, outlier_log = handle_outliers(X_train, X_test)
            process_log.append(outlier_log)

            X_train, X_test, encode_log = encode_categorical(X_train, X_test)
            if encode_log:
                process_log.append(encode_log)

            feature_columns = list(X_train.columns)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            process_log.append({
                "step": "Data scaled",
                "details": [
                    "Scaler used: StandardScaler",
                    f"Numeric features scaled: {len(feature_columns)}"
                ]
            })

            X_train = np.hstack([X_train, X_text_train])
            X_test = np.hstack([X_test, X_text_test])

        else:
            X_train = X_text_train
            X_test = X_text_test
            feature_columns = [f"tfidf_{i}" for i in range(X_train.shape[1])]

    else:

        X_train, X_test, missing_log = handle_missing(X_train, X_test)
        process_log.append(missing_log)
        
        X_train, X_test, outlier_log = handle_outliers(X_train, X_test)
        process_log.append(outlier_log)

        X_train, X_test, encode_log = encode_categorical(X_train, X_test)
        if encode_log:
            process_log.append(encode_log)

        feature_columns = list(X_train.columns)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        process_log.append({
            "step": "Data scaled",
            "details": [
                "Scaler used: StandardScaler",
                f"Numeric features scaled: {len(feature_columns)}"
            ]
        })


    if X_train.shape[1] < 2:
        raise ValueError(" Too many columns removed! Check preprocessing.")

    return X_train, X_test, y_train, y_test, {
        "vectorizer": vectorizer,
        "text_cols": text_cols,
        "scaler": scaler,
        "columns": feature_columns,
        "raw_columns": raw_columns,
        "raw_dtypes": raw_dtypes,
        "raw_mins": raw_mins,
        "raw_maxes": raw_maxes,
        "categorical_values": categorical_values,
        "process_log": process_log
    }