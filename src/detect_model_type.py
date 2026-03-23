import pandas as pd
def detect_model_type(df, target_column, threshold=20):
    target = df[target_column]

    if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return "classification"

    if pd.api.types.is_numeric_dtype(target):
        if target.nunique() < threshold:
            return "classification"
        else:
            return "regression"

    return "classification"