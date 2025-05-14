# Re-import necessary packages after environment reset

import os
import numpy as np
import joblib
from evaluate_classifier import evaluate_classifier

# -------------------------
# Config
# -------------------------
FEATURE_DIR = "features"
MODEL_DIR = "models"
EVAL_TYPE = "visual"
INPUT_NAME = "real_test_on_pretrained_dcsass_ravdess"

FER_PATH = os.path.join(FEATURE_DIR, "real_ravdess_visual_features.npy")
RESNET_PATH = os.path.join(FEATURE_DIR, "real_dcsass_visual_features.npy")
LABELS_PATH = os.path.join(FEATURE_DIR, "real_ravdess_visual_labels.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_classifier_pretrained.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_scaler_pretrained.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_label_encoder_pretrained.pkl")

# -------------------------
# Load data
# -------------------------
if os.path.exists(FER_PATH) and os.path.exists(RESNET_PATH) and os.path.exists(LABELS_PATH):
    fer = np.load(FER_PATH)
    resnet = np.load(RESNET_PATH)
    labels = np.load(LABELS_PATH)

    min_len = min(len(fer), len(resnet), len(labels))
    fer = fer[:min_len]
    resnet = resnet[:min_len]
    labels = labels[:min_len]

    X_real = np.hstack([fer, resnet])
    y_real = labels

    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    X_real_scaled = scaler.transform(X_real)
    y_real_encoded = label_encoder.transform(y_real)
    y_pred = clf.predict(X_real_scaled)

    evaluate_classifier(
        y_true=y_real_encoded,
        y_pred=y_pred,
        label_encoder=label_encoder,
        modality=EVAL_TYPE,
        input_name=INPUT_NAME,
    )

else:
    print("❌ Required feature or label files not found.")
