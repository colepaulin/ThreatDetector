import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

def train_and_evaluate_audio_classifier(features_path="features/audio_features.npy", labels_path="features/audio_labels.npy"):
    # Load features and labels
    X = np.load(features_path)
    y = np.load(labels_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a logistic regression model
    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        modality="audio",       # or "audio", "vision"
        input_name="emotion_classifier_baseline"  # 🏷️ custom tag
    )

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")

    # Save model and scalers
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/audio_emotion_classifier.pkl")
    joblib.dump(scaler, "models/audio_emotion_scaler.pkl")
    joblib.dump(label_encoder, "models/audio_emotion_label_encoder.pkl")
    print("✅ Model, scaler, and label encoder saved to `models/`")

if __name__ == "__main__":
    train_and_evaluate_audio_classifier()
