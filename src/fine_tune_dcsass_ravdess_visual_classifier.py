import numpy as np
import os
import joblib
from sklearn.metrics import classification_report
from evaluate_classifier import evaluate_classifier

# ------------------
# Paths
# ------------------
MODEL_DIR = "models"
FEATURE_DIR = "features"

# Load real individual modality features
fer = np.load(os.path.join(FEATURE_DIR, "real_ravdess_visual_features.npy"))
resnet = np.load(os.path.join(FEATURE_DIR, "real_dcsass_visual_features.npy"))
labels = np.load(os.path.join(FEATURE_DIR, "real_ravdess_visual_labels.npy"))

# Align by length
min_len = min(len(fer), len(resnet), len(labels))
fer = fer[:min_len]
resnet = resnet[:min_len]
labels = labels[:min_len]

# Fuse features
X_real = np.hstack([fer, resnet])
y_real = labels

# Load pretrained classifier and preprocessing
clf = joblib.load(os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_classifier_pretrained.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_scaler_pretrained.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_label_encoder_pretrained.pkl"))

# Scale and encode
X_real_scaled = scaler.transform(X_real)
y_real_encoded = label_encoder.transform(y_real)

# Fine-tune classifier
clf.set_params(warm_start=True, max_iter=1)
for _ in range(10):
    clf.fit(X_real_scaled, y_real_encoded)

# Predict and evaluate
y_pred = clf.predict(X_real_scaled)

# Run evaluation
print("📊 Fine-tuned on real fused visual data:")
evaluate_classifier(y_real_encoded, y_pred, label_encoder, modality="visual", input_name="fused_finetuneandeval_on_real")

# Save model
joblib.dump(clf, os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_classifier_finetuned.pkl"))
print("✅ Saved fine-tuned model to models/visual_intent_classifier_finetuned.pkl")
