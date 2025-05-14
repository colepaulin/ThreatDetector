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

# Load real data
X_real = np.load(os.path.join(FEATURE_DIR, "real_ravdess_visual_features.npy"))
y_real = np.load(os.path.join(FEATURE_DIR, "real_ravdess_visual_labels.npy"))

# Load pre-trained classifier and preprocessing components
clf = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_classifier.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "ravdess_visual_label_encoder.pkl"))

# Transform real data
X_real_scaled = scaler.transform(X_real)
y_real_encoded = label_encoder.transform(y_real)

# Enable warm start and low iteration count for safe fine-tuning
clf.set_params(warm_start=True, max_iter=1)

# Fine-tune for N small steps
n_epochs = 70
for _ in range(n_epochs):
    clf.fit(X_real_scaled, y_real_encoded)

# Evaluate on real data
y_pred = clf.predict(X_real_scaled)
print("📊 Fine-tuned on real visual data:")
evaluate_classifier(y_real_encoded,
                    y_pred=y_pred,
                    label_encoder=label_encoder,
                    modality="visual",
                    input_name="ravdess_finetuneandeval_on_real")
# Save fine-tuned model
joblib.dump(clf, os.path.join(MODEL_DIR, "ravdess_visual_intent_classifier_finetuned.pkl"))
