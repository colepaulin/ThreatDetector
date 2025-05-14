# Re-run script now that environment has been reset

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from evaluate_classifier import evaluate_classifier

# -------------------------
# Config
# -------------------------
FEATURE_DIR = "features"
MODEL_DIR = "models"
EVAL_TYPE = "visual"
INPUT_NAME = "pretrain_ravdess_dcsass"

FER_PATH = os.path.join(FEATURE_DIR, "ravdess_visual_features.npy")
FER_LABELS = os.path.join(FEATURE_DIR, "ravdess_visual_labels.npy")
RESNET_PATH = os.path.join(FEATURE_DIR, "dcsass_visual_features.npy")
RESNET_LABELS = os.path.join(FEATURE_DIR, "dcsass_visual_labels.npy")

# -------------------------
# Load and align data
# -------------------------
if os.path.exists(FER_PATH) and os.path.exists(RESNET_PATH):
    fer = np.load(FER_PATH)
    fer_labels = np.load(FER_LABELS)
    resnet = np.load(RESNET_PATH)
    resnet_labels = np.load(RESNET_LABELS)

    fer_resnet_zeros = np.zeros((len(fer), resnet.shape[1]))
    resnet_fer_zeros = np.zeros((len(resnet), fer.shape[1]))

    fer_fused = np.hstack([fer, fer_resnet_zeros])
    resnet_fused = np.hstack([resnet_fer_zeros, resnet])

    X = np.vstack([fer_fused, resnet_fused])
    y = np.concatenate([fer_labels, resnet_labels])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    evaluate_classifier(
        y_true=y_test,
        y_pred=clf.predict(X_test),
        label_encoder=label_encoder,
        modality=EVAL_TYPE,
        input_name=INPUT_NAME,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_classifier_pretrained.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_scaler_pretrained.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "dcsass_ravdess_visual_intent_label_encoder_pretrained.pkl"))
else:
    print("❌ Required feature files not found. Please ensure ravdess and dcsass visual features are saved.")
