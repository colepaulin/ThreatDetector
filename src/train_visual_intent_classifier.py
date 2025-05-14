import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

# Paths

def train_visual_intent_classifier(feature_path = "features/visual_features.npy", label_path = "features/visual_labels.npy", models_dir = "models"):
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load data
    X = np.load(feature_path)
    y = np.load(label_path)

    # 2. Encode labels (friendly/threat → 0/1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # 5. Train MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        modality="visual",       # or "audio", "vision"
        input_name="ravdess_evaluated_on_ravdess"  # 🏷️ custom tag
    )

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    
    # 7. Save model + helpers
    joblib.dump(clf, os.path.join(models_dir, "visual_intent_classifier.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "visual_intent_scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "visual_intent_label_encoder.pkl"))

    print("\n✅ Saved model, scaler, and label encoder to /models/")

if __name__ == "__main__":
    train_visual_intent_classifier()