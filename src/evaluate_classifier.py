import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def evaluate_classifier(y_true, y_pred, label_encoder, modality="fused", input_name="default_run"):
    # -------------------------------
    # Prepare paths
    # -------------------------------
    out_dir = os.path.join("evaluation", modality, input_name)
    os.makedirs(out_dir, exist_ok=True)

    labels = label_encoder.classes_
    y_true_decoded = label_encoder.inverse_transform(y_true)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # -------------------------------
    # Metrics Report
    # -------------------------------
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True
    )

    # Add additional metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")

    report["overall"] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }

    # Save report as JSON
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"📊 Saved metrics JSON to: {json_path}")

    # -------------------------------
    # Confusion Matrix Plot
    # -------------------------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{modality.upper()} Confusion Matrix")

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"🖼️ Saved confusion matrix to: {cm_path}")
