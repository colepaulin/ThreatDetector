import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

# -------------------
# Config
# -------------------
FEATURE_DIR = "features"
MODEL_DIR = "models"
EVAL_DIR = "evaluation/audio/real_data_evaluated_on_ravdess"
os.makedirs(EVAL_DIR, exist_ok=True)

X_REAL_PATH = os.path.join(FEATURE_DIR, "real_ravdess_audio_features.npy")
Y_REAL_PATH = os.path.join(FEATURE_DIR, "real_ravdess_audio_labels.npy")

# -------------------
# Load Real Features + Labels
# -------------------
X_real = np.load(X_REAL_PATH)
y_real = np.load(Y_REAL_PATH)

# -------------------
# Load DCSASS-trained Classifier
# -------------------
clf = joblib.load(os.path.join(MODEL_DIR, "ravdess_audio_classifier.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "ravdess_audio_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "ravdess_audio_label_encoder.pkl"))

# -------------------
# Preprocess Real Data
# -------------------
X_real_scaled = scaler.transform(X_real)
y_real_encoded = label_encoder.transform(y_real)

# -------------------
# Predict and Evaluate
# -------------------
y_pred = clf.predict(X_real_scaled)
acc = accuracy_score(y_real_encoded, y_pred)
print(f"✅ Accuracy on real-world data: {acc:.4f}")


evaluate_classifier(
    y_true=y_real_encoded,
    y_pred=y_pred,
    label_encoder=label_encoder,
    modality="visual",       # or "audio", "vision"
    input_name="real_data_evaluated_on_ravdess"  # 🏷️ custom tag
)
