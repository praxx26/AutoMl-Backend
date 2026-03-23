import numpy as np
from joblib import Parallel, delayed
import time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


# 🔥 GLOBAL TIME LIMIT (avoid Render timeout)
MAX_TIME = 60  # seconds


def process_model(name, model, param_grid,
                  X_train, X_test, y_train, y_test,
                  task, cv_splits, n_iter):

    print(f"\n🚀 Training {name}...")

    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        cv = cv_splits
        scoring = "r2"

    grid = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,                 # 🔥 reduced iterations
        cv=cv,
        scoring=scoring,
        n_jobs=1,                      # 🔥 CRITICAL FIX
        random_state=42
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    cv_score = grid.best_score_

    # 🔥 Fit status detection
    if train_score - test_score > 0.1:
        fit_status = "Overfitting"
    elif train_score < 0.7 and test_score < 0.7:
        fit_status = "Underfitting"
    else:
        fit_status = "Good Fit"

    print(f"📊 {name} → {fit_status}")

    return {
        "name": name,
        "model": best_model,
        "test_score": test_score,
        "cv_score": cv_score,
        "train_score": train_score,
        "params": grid.best_params_,
        "fit_status": fit_status
    }


def train_best_model(X_train, X_test, y_train, y_test):

    start_time = time.time()

    # 🔍 Detect task
    if y_train.dtype == "object" or len(np.unique(y_train)) <= 10:
        task = "classification"
    else:
        task = "regression"

    print("🔍 Detected Task:", task)

    label_encoder = None
    if task == "classification" and y_train.dtype == "object":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    # 🔥 REDUCED PARAM SPACE (still AutoML but lighter)
    if task == "classification":
        models = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"C": [0.1, 1, 10], "solver": ["lbfgs"]}
            ),

            "KNN": (
                KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7]}
            ),

            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {"n_estimators": [50, 100], "max_depth": [None, 10]}
            ),

            "Gradient Boosting": (
                GradientBoostingClassifier(),
                {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            ),

            "SVM": (
                SVC(),
                {"C": [0.1, 1], "kernel": ["rbf"]}
            )
        }

    else:
        models = {
            "Ridge": (
                Ridge(),
                {"alpha": [0.1, 1, 10]}
            ),

            "KNN": (
                KNeighborsRegressor(),
                {"n_neighbors": [3, 5]}
            ),

            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {"n_estimators": [50, 100], "max_depth": [None, 10]}
            ),

            "Gradient Boosting": (
                GradientBoostingRegressor(),
                {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            ),

            "SVR": (
                SVR(),
                {"C": [0.1, 1]}
            )
        }

    results = []

    # 🔥 SEQUENTIAL (instead of Parallel)
    for name, (model, params) in models.items():

        # ⛔ Stop if time exceeded
        if time.time() - start_time > MAX_TIME:
            print("⏱️ Time limit reached, stopping training...")
            break

        try:
            result = process_model(
                name, model, params,
                X_train, X_test, y_train, y_test,
                task, cv_splits=3, n_iter=5   # 🔥 reduced
            )
            results.append(result)

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    if not results:
        raise Exception("No models trained successfully")

    best_result = max(results, key=lambda x: x["test_score"])

    print("\n==============================")
    print("🏆 BEST MODEL:", best_result["name"])
    print("Test Score:", best_result["test_score"])
    print("CV Score:", best_result["cv_score"])
    print("Train Score:", best_result["train_score"])
    print("Fit Status:", best_result["fit_status"])
    print("==============================")

    return {
        "model": best_result["model"],
        "best_model_name": best_result["name"],
        "best_params": best_result["params"],
        "label_encoder": label_encoder,
        "all_results": results
    }