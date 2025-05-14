import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

# Mapping from emotion to intent
emotion_to_intent = {
    "happy": "friendly",
    "calm": "friendly",
    "neutral": "neutral",
    "angry": "threat",
    "fearful": "threat",
    "disgust": "threat",
    "sad": "friendly",       # include this if you want 3-class
    "surprised": "friendly"
}

def train_audio_intent_classifier(features_path="features/ravdess_audio_features.npy", labels_path="features/ravdess_audio_labels.npy", num_classes=2):
    # Load features and labels
    X = np.load(features_path)
    y_emotions = np.load(labels_path)

    # Convert emotion labels → intent labels
    y_intent = np.array([emotion_to_intent[label] for label in y_emotions])

    # Reduce to 2 classes if specified
    if num_classes == 2:
        keep = np.isin(y_intent, ["friendly", "threat"])
        X = X[keep]
        y_intent = y_intent[keep]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_intent)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train MLP
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        modality="audio",       # or "audio", "vision"
        input_name="ravdess_evaluated_on_ravdess"  # 🏷️ custom tag
    )

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/ravdess_audio_intent_classifier.pkl")
    joblib.dump(scaler, "models/ravdess_audio_intent_scaler.pkl")
    joblib.dump(label_encoder, "models/ravdess_audio_intent_label_encoder.pkl")
    print("✅ Intent model, scaler, and label encoder saved to `models/`")

if __name__ == "__main__":
    train_audio_intent_classifier()  # Set to 3 for 3-class mode
