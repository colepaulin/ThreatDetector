import numpy as np
import joblib
import os
from sklearn.metrics import classification_report
from evaluate_classifier import evaluate_classifier

# --------------------
# Config
# --------------------
FEATURE_DIR = "features"
MODEL_DIR = "models"

# Real-world features and labels
AUDIO_FEATURES_PATH = os.path.join(FEATURE_DIR, "real_audio_features.npy")
VISUAL_FEATURES_PATH = os.path.join(FEATURE_DIR, "real_visual_features.npy")
LABELS_PATH = os.path.join(FEATURE_DIR, "real_audio_labels.npy")  # or visual_labels.npy (should match)

# --------------------
# Load Data
# --------------------
audio_features = np.load(AUDIO_FEATURES_PATH)
visual_features = np.load(VISUAL_FEATURES_PATH)
labels = np.load(LABELS_PATH)

# Align sizes just in case
min_len = min(len(audio_features), len(visual_features), len(labels))
audio_features = audio_features[:min_len]
visual_features = visual_features[:min_len]
labels = labels[:min_len]

# --------------------
# Fuse Features
# --------------------
fused_features = np.hstack([audio_features, visual_features])

# --------------------
# Load preprocessing tools
# --------------------
scaler = joblib.load(os.path.join(MODEL_DIR, "fused_intent_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "fused_intent_label_encoder.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "fused_intent_classifier.pkl"))

# --------------------
# Transform features and labels
# --------------------
X_real_scaled = scaler.transform(fused_features)
y_real_encoded = label_encoder.transform(labels)

# --------------------
# Fine-tune classifier (warm start)
# --------------------
clf.set_params(warm_start=True, max_iter=1)

for _ in range(30):
    clf.fit(X_real_scaled, y_real_encoded)

# --------------------
# Evaluate on same data
# --------------------
y_pred = clf.predict(X_real_scaled)
print("📊 Fine-tuned on real fused data:")
evaluate_classifier(y_real_encoded,
                    y_pred=y_pred,
                    label_encoder=label_encoder,
                    modality="fused",
                    input_name="real_fused_intent_classifier_baseline")
# --------------------
# Save fine-tuned model
# --------------------
joblib.dump(clf, os.path.join(MODEL_DIR, "fused_intent_classifier_finetuned.pkl"))
print("✅ Fine-tuned fused classifier saved to models/fused_intent_classifier_finetuned.pkl")
