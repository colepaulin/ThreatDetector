import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

# --------------------
# Config
# --------------------
FEATURE_DIR = "features"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_FEATURES_PATH = os.path.join(FEATURE_DIR, "audio_features.npy")
VISUAL_FEATURES_PATH = os.path.join(FEATURE_DIR, "visual_features.npy")
LABELS_PATH = os.path.join(FEATURE_DIR, "visual_labels.npy")  # Labels align with Actor 1 only

# --------------------
# Load Data
# --------------------
audio_features = np.load(AUDIO_FEATURES_PATH)
visual_features = np.load(VISUAL_FEATURES_PATH)
labels = np.load(LABELS_PATH)

# --------------------
# Align by actor (truncate to smallest set)
# --------------------
min_len = min(len(audio_features), len(visual_features), len(labels))
audio_features = audio_features[:min_len]
visual_features = visual_features[:min_len]
labels = labels[:min_len]

# --------------------
# Concatenate Features
# --------------------
fused_features = np.hstack([audio_features, visual_features])

# --------------------
# Encode Labels
# --------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# --------------------
# Scale Features
# --------------------
scaler = StandardScaler()
fused_scaled = scaler.fit_transform(fused_features)

# --------------------
# Train/Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    fused_scaled, y_encoded, test_size=0.2, random_state=42
)

# --------------------
# Train Classifier
# --------------------
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

# --------------------
# Evaluate
# --------------------
y_pred = clf.predict(X_test)
evaluate_classifier(
    y_true=y_test,
    y_pred=y_pred,
    label_encoder=label_encoder,
    modality="fused",       # or "audio", "vision"
    input_name="intent_classifier_baseline"  # 🏷️ custom tag
)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}\n")

# --------------------
# Save Model
# --------------------
joblib.dump(clf, os.path.join(OUTPUT_DIR, "fused_intent_classifier.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "fused_intent_scaler.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "fused_intent_label_encoder.pkl"))

print(f"\n✅ Saved fused model and preprocessing tools to '{OUTPUT_DIR}/'")
