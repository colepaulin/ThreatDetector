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
EVAL_DIR = "evaluation/dcsass_on_real"
os.makedirs(EVAL_DIR, exist_ok=True)

X_REAL_PATH = os.path.join(FEATURE_DIR, "real_ravdess_visual_features.npy")
Y_REAL_PATH = os.path.join(FEATURE_DIR, "real_ravdess_visual_labels.npy")

# -------------------
# Load Real Features + Labels
# -------------------
X_real = np.load(X_REAL_PATH)
y_real = np.load(Y_REAL_PATH)

# -------------------
# Load DCSASS-trained Classifier
# -------------------
clf = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_classifier.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_label_encoder.pkl"))

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
# print("📊 Classification Report (DCSASS on Real Visual Data):")
# print(classification_report(y_real_encoded, y_pred, target_names=label_encoder.classes_))

# -------------------
# Save Confusion Matrix
# -------------------
# cm = confusion_matrix(y_real_encoded, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("DCSASS Visual Classifier on Real Visual Data")
# plt.tight_layout()
# plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
# plt.close()

evaluate_classifier(
    y_true=y_real_encoded,
    y_pred=y_pred,
    label_encoder=label_encoder,
    modality="visual",       # or "audio", "vision"
    input_name="real_data_evaluated_on_ravdess"  # 🏷️ custom tag
)
