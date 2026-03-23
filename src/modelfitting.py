import numpy as np
from joblib import Parallel, delayed

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


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
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        random_state=42
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    cv_score = grid.best_score_

    if train_score - test_score > 0.1:
        fit_status = "Overfitting"
    elif train_score < 0.7 and test_score < 0.7:
        fit_status = "Underfitting"
    else:
        fit_status = "Good Fit"

    print(f"📊 {name} → {fit_status}")

    if fit_status != "Good Fit":

        print(f"⚠️ Adjusting {name}...")

        new_params = param_grid.copy()

    if fit_status == "Overfitting":

        print("⚠️ Fixing Overfitting...")

        if "max_depth" in new_params:
            new_params["max_depth"] = [3, 5, 10]

        if "min_samples_split" in new_params:
            new_params["min_samples_split"] = [5, 10]

        if "min_samples_leaf" in new_params:
            new_params["min_samples_leaf"] = [2, 4]


        if "C" in new_params:
            new_params["C"] = [0.001, 0.01, 0.1]

        if "learning_rate" in new_params:
            new_params["learning_rate"] = [0.01, 0.05]

        if "n_neighbors" in new_params:
            new_params["n_neighbors"] = [7, 9, 11]


    elif fit_status == "Underfitting":

        print("⚠️ Fixing Underfitting...")

   
        if "max_depth" in new_params:
            new_params["max_depth"] = [None, 20, 30]

        if "min_samples_split" in new_params:
            new_params["min_samples_split"] = [2, 3]

        if "min_samples_leaf" in new_params:
            new_params["min_samples_leaf"] = [1]

        

    
        if "C" in new_params:
            new_params["C"] = [1, 10, 100]

    
        if "learning_rate" in new_params:
            new_params["learning_rate"] = [0.1, 0.2]

        if "n_neighbors" in new_params:
            new_params["n_neighbors"] = [3, 5]

        print(f"🔄 Retuning {name} with adjusted params...")

        grid = RandomizedSearchCV(
            model,
            param_distributions=new_params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            random_state=42
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        cv_score = grid.best_score_

        if train_score - test_score > 0.1:
            fit_status = "Overfitting"
        elif train_score < 0.7 and test_score < 0.7:
            fit_status = "Underfitting"
        else:
            fit_status = "Good Fit"

        print(f"✅ After tuning → {fit_status}")

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

    if task == "classification":
        models = {
            "Logistic Regression": (
                LogisticRegression(max_iter=2000),
                {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"]}
            ),

            "KNN": (
                KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
            ),

            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [50],
                    "max_depth": [None, 10, 20]
                }
            ),

            "Gradient Boosting": (
                GradientBoostingClassifier(),
                {
                    "n_estimators": [50],
                    "learning_rate": [0.01, 0.1]
                }
            ),

            "SVM": (
                SVC()
            )
        }

    else:
        models = {
            "Ridge": (
                Ridge(),
                {"alpha": [0.01, 0.1, 1, 10]}
            ),

            "KNN": (
                KNeighborsRegressor(),
                {"n_neighbors": [3, 5, 7]}
            ),

            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": [50],
                    "max_depth": [None, 10]
                }
            ),

            "Gradient Boosting": (
                GradientBoostingRegressor(),
                {
                    "n_estimators": [50],
                    "learning_rate": [0.01, 0.1]
                }
            ),

            "SVR": (
                SVR()
            )
        }

    results = Parallel(n_jobs=1)(
        delayed(process_model)(
            name, model, params,
            X_train, X_test, y_train, y_test,
            task, cv_splits=3, n_iter=5
        )
        for name, (model, params) in models.items()
    )

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