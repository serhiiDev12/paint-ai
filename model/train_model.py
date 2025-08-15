import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import json

FEEDBACK_PATH = Path("feedback/feedback_log.json")

def load_feedback():
    if not FEEDBACK_PATH.exists():
        return [], []
    with open(FEEDBACK_PATH) as f:
        log = json.load(f)

    X = []
    y = []

    for item in log:
        img = cv2.imread(item["output"])
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        features = img.flatten() / 255.0
        X.append(features)
        y.append(1 if item["feedback"] == "good" else 0)

    return np.array(X), np.array(y)

def train_classifier(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)
    print(f"Validation Accuracy: {acc:.2f}")
    return clf

def main():
    X, y = load_feedback()
    if len(X) < 10:
        print("Not enough data to train.")
        return
    model = train_classifier(X, y)
    joblib.dump(model, "model/quality_predictor.pkl")
    print("Model saved.")

if __name__ == "__main__":
    main()