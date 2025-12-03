# ml_models.py
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ------------------------
# Helper: flatten signals for ML
# ------------------------
def prepare_features(X):
    """
    Flatten (num_samples, timesteps[, channels]) -> (num_samples, features)
    Example: (500, 1000) -> (500, 1000)
    """
    return X.reshape(X.shape[0], -1)


# ------------------------
# Define classical ML models
# ------------------------
def get_ml_models(num_classes):
    models = {
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, 
                   class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean"),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", num_class=num_classes,
            n_jobs=-1, random_state=42, reg_alpha=0.1, reg_lambda=0.1
        )
    }
    return models


# ------------------------
# Train + Evaluate ML model
# ------------------------
def train_ml_model(name, model, X_train, y_train, X_val, y_val):
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    print(f"✅ {name} accuracy: {acc:.4f}")
    return model, acc
