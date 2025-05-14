import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_classifier import evaluate_classifier

# -------------------
# Config
# -------------------
FEATURE_DIR = "features"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(FEATURE_DIR, "dcsass_visual_features.npy")
LABELS_PATH = os.path.join(FEATURE_DIR, "dcsass_visual_labels.npy")

# -------------------
# Load Data
# -------------------
X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# -------------------
# Encode Labels
# -------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------
# Scale Features
# -------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------
# Train/Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# -------------------
# Train Classifier
# -------------------
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

# -------------------
# Evaluate
# -------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("📊 Classification Report (DCSASS Visual Classifier):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DCSASS Visual Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation/dcsass_visual_confusion_matrix.png")
plt.close()
evaluate_classifier(
    y_true=y_test,
    y_pred=y_pred,
    label_encoder=label_encoder,
    modality="visual",       # or "audio", "vision"
    input_name="dcsass_intent_classifier_baseline"  # 🏷️ custom tag
)

# -------------------
# Save Model
# -------------------
joblib.dump(clf, os.path.join(OUTPUT_DIR, "dcsass_visual_classifier.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "dcsass_visual_scaler.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "dcsass_visual_label_encoder.pkl"))
print("✅ Saved classifier, scaler, and label encoder.")
