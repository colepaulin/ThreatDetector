import os
import numpy as np
import joblib
from sklearn.metrics import classification_report
from evaluate_classifier import evaluate_classifier  # Make sure this is in your PYTHONPATH

def weighted_vote_fusion(
    weights=(0.25, 0.25, 0.25, 0.25),

    real_ravdess_audio_features="features/real_ravdess_audio_features.npy",
    real_ravdess_visual_features="features/real_ravdess_visual_features.npy",
    real_dcsass_visual_features="features/real_dcsass_visual_features.npy",

    real_ravdess_labels="features/real_ravdess_visual_labels.npy",

    ravdess_audio_classifier_finetuned="models/ravdess_audio_classifier_finetuned.pkl",
    ravdess_audio_scaler="models/ravdess_audio_scaler.pkl",
    ravdess_audio_label_encoder="models/ravdess_audio_label_encoder.pkl",

    ravdess_visual_classifier_finetuned="models/ravdess_visual_classifier_finetuned.pkl",
    ravdess_visual_scaler="models/ravdess_visual_scaler.pkl",
    ravdess_visual_label_encoder="models/ravdess_visual_label_encoder.pkl",

    dcsass_visual_classifier="models/dcsass_visual_classifier_finetuned.pkl",
    dcsass_visual_scaler="models/dcsass_visual_scaler.pkl",
    dcsass_visual_label_encoder="models/dcsass_visual_label_encoder.pkl",

    dcsass_ravdess_visual_classifier="models/dcsass_ravdess_visual_classifier_finetuned.pkl",
    dcsass_ravdess_visual_scaler="models/dcsass_ravdess_visual_scaler_pretrained.pkl",
    dcsass_ravdess_visual_label_encoder="models/dcsass_ravdess_visual_label_encoder_pretrained.pkl",
):
    modality_name=f"w={weights}"
    real_ravdess_audio_features = np.load(real_ravdess_audio_features)
    real_ravdess_visual_features = np.load(real_ravdess_visual_features)
    real_dcsass_visual_features = np.load(real_dcsass_visual_features)
    real_dcsass_ravdess_visual_features = np.hstack([real_ravdess_visual_features, real_dcsass_visual_features])
    real_ravdess_labels = np.load(real_ravdess_labels)

    min_len = min(len(real_ravdess_audio_features), 
                  len(real_ravdess_visual_features), 
                  len(real_dcsass_visual_features), 
                  len(real_dcsass_ravdess_visual_features),
                  len(real_ravdess_labels))
    
    real_ravdess_audio_features = real_ravdess_audio_features[:min_len]
    real_ravdess_visual_features = real_ravdess_visual_features[:min_len]
    real_dcsass_visual_features = real_dcsass_visual_features[:min_len]
    real_dcsass_ravdess_visual_features = real_dcsass_ravdess_visual_features[:min_len]
    real_ravdess_labels = real_ravdess_labels[:min_len]

    ravdess_audio_classifier_finetuned = joblib.load(ravdess_audio_classifier_finetuned)
    ravdess_audio_scaler = joblib.load(ravdess_audio_scaler)
    ravdess_audio_label_encoder = joblib.load(ravdess_audio_label_encoder)

    ravdess_visual_classifier_finetuned = joblib.load(ravdess_visual_classifier_finetuned)
    ravdess_visual_scaler = joblib.load(ravdess_visual_scaler)
    ravdess_visual_label_encoder = joblib.load(ravdess_visual_label_encoder)

    dcsass_visual_classifier = joblib.load(dcsass_visual_classifier)
    dcsass_visual_scaler = joblib.load(dcsass_visual_scaler)
    dcsass_visual_label_encoder = joblib.load(dcsass_visual_label_encoder)

    dcsass_ravdess_visual_classifier = joblib.load(dcsass_ravdess_visual_classifier)
    dcsass_ravdess_visual_scaler = joblib.load(dcsass_ravdess_visual_scaler)
    dcsass_ravdess_visual_label_encoder = joblib.load(dcsass_ravdess_visual_label_encoder)

    audio_proba = ravdess_audio_classifier_finetuned.predict_proba(ravdess_audio_scaler.transform(real_ravdess_audio_features))
    ravdess_visual_proba = ravdess_visual_classifier_finetuned.predict_proba(ravdess_visual_scaler.transform(real_ravdess_visual_features))
    dcsass_visual_proba = dcsass_visual_classifier.predict_proba(dcsass_visual_scaler.transform(real_dcsass_visual_features))
    dcsass_ravdess_visual_proba = dcsass_ravdess_visual_classifier.predict_proba(dcsass_ravdess_visual_scaler.transform(real_dcsass_ravdess_visual_features))

    assert list(ravdess_audio_label_encoder.classes_) == list(ravdess_visual_label_encoder.classes_) == list(dcsass_visual_label_encoder.classes_) == list(dcsass_ravdess_visual_label_encoder.classes_)

    w_audio, w_ravdess_visual, w_dcsass_visual, w_dcsass_ravdess_visual = weights

    fusion_modes = {
        "ravdess_visual": (w_audio, w_ravdess_visual, 0.0, 0.0),
        "dcsass_visual": (w_audio, 0.0, w_dcsass_visual, 0.0),
        "dcsass_ravdess_cat_visual": (w_audio, 0, 0, w_dcsass_ravdess_visual),
        "full": (w_audio, w_ravdess_visual, w_dcsass_visual, 0.0),
    }

    for name, (wa, wrv, wdv, wdrv) in fusion_modes.items():
        fused_proba = wa * audio_proba + wrv * ravdess_visual_proba + wdv * dcsass_visual_proba + wdrv * dcsass_ravdess_visual_proba
        y_pred = np.argmax(fused_proba, axis=1)
        y_true_enc = ravdess_audio_label_encoder.transform(real_ravdess_labels)

        print(f"\n📊 Fusion Mode: {name} (Weights: A={wa:.2f}, R={wrv:.2f}, D={wdv:.2f}, DR={wdrv:.2f})")
        evaluate_classifier(
            y_true=y_true_enc,
            y_pred=y_pred,
            label_encoder=ravdess_audio_label_encoder,
            modality="fusion",
            input_name=f"{name}",
        )

if __name__ == "__main__":
    weighted_vote_fusion(weights=(0.1, 0.1, 0.1, 0.7))
