import os
import numpy as np
import joblib
import cv2
from fer import FER
from moviepy.editor import VideoFileClip
from run_wav2vec2 import extract_wav2vec2_features
from tempfile import NamedTemporaryFile
import sys
import torch
from torchvision import transforms
from torchvision.models import resnet50

# ----------------------------
# Extract audio from video using moviepy
# ----------------------------
def extract_audio_from_mp4(mp4_path, target_rate=16000):
    temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = temp_file.name
    temp_file.close()

    video = VideoFileClip(mp4_path)
    video.audio.write_audiofile(wav_path, fps=target_rate, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
    return wav_path

# ----------------------------
# Extract FER visual features
# ----------------------------
def extract_fer_features_from_video(video_path, frame_skip=5):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        try:
            result = detector.detect_emotions(frame)
            if result:
                emotions = result[0]["emotions"]
                vec = [emotions[e] for e in sorted(emotions)]
                frame_features.append(vec)
        except:
            continue

    cap.release()
    return np.mean(frame_features, axis=0) if frame_features else np.zeros(7)

# ----------------------------
# Extract ResNet visual features
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_action_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // 20)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            try:
                img = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img).squeeze().cpu().numpy()
                    frame_features.append(feat)
            except:
                continue
        idx += 1

    cap.release()
    return np.mean(frame_features, axis=0) if frame_features else np.zeros(2048)

# (Keep all previous imports and feature extraction code unchanged)

# ----------------------------
# Full Prediction Pipeline
# ----------------------------
def predict_intent_from_mp4(mp4_path):
    print(f"📂 Predicting intent from: {mp4_path}")

    # 1. Extract features
    wav_path = extract_audio_from_mp4(mp4_path)
    audio_feat = extract_wav2vec2_features(wav_path).reshape(1, -1)
    os.remove(wav_path)

    rav_visual_feat = extract_fer_features_from_video(mp4_path).reshape(1, -1)
    dc_visual_feat = extract_action_features(mp4_path).reshape(1, -1)
    dc_rav_visual_feat = np.hstack([rav_visual_feat, dc_visual_feat])
    # audio_rav_fused_feat = np.hstack([audio_feat, rav_visual_feat])

    # 2. Load classifiers
    audio_clf = joblib.load("models/ravdess_audio_classifier_finetuned.pkl")
    audio_scaler = joblib.load("models/ravdess_audio_scaler.pkl")
    audio_encoder = joblib.load("models/ravdess_audio_label_encoder.pkl")

    rav_visual_clf = joblib.load("models/ravdess_visual_classifier_finetuned.pkl")
    rav_visual_scaler = joblib.load("models/ravdess_visual_scaler.pkl")
    rav_visual_encoder = joblib.load("models/ravdess_visual_label_encoder.pkl")

    dc_visual_clf = joblib.load("models/dcsass_visual_classifier_finetuned.pkl")
    dc_visual_scaler = joblib.load("models/dcsass_visual_scaler.pkl")
    dc_visual_encoder = joblib.load("models/dcsass_visual_label_encoder.pkl")

    dc_rav_visual_clf = joblib.load("models/dcsass_ravdess_visual_classifier_finetuned.pkl")
    dc_rav_visual_scaler = joblib.load("models/dcsass_ravdess_visual_scaler_pretrained.pkl")
    dc_rav_visual_encoder = joblib.load("models/dcsass_ravdess_visual_label_encoder_pretrained.pkl")

    # fused_clf = joblib.load("models/fused_intent_classifier_finetuned.pkl")
    # fused_scaler = joblib.load("models/fused_intent_scaler.pkl")
    # fused_encoder = joblib.load("models/fused_intent_label_encoder.pkl")



    # 3. Scale & predict from each model
    audio_scaled = audio_scaler.transform(audio_feat)
    rav_visual_scaled = rav_visual_scaler.transform(rav_visual_feat)
    dc_visual_scaled = dc_visual_scaler.transform(dc_visual_feat)
    dc_rav_visual_scaled = dc_rav_visual_scaler.transform(dc_rav_visual_feat)

    audio_pred = audio_encoder.inverse_transform(
        audio_clf.predict(audio_scaled)
    )[0]

    rav_visual_pred = rav_visual_encoder.inverse_transform(
        rav_visual_clf.predict(rav_visual_scaled)
    )[0]

    dc_visual_pred = dc_visual_encoder.inverse_transform(
        dc_visual_clf.predict(dc_visual_scaled)
    )[0]

    dc_rav_visual_pred = dc_rav_visual_encoder.inverse_transform(
        dc_rav_visual_clf.predict(dc_rav_visual_scaled)
    )[0]

    # Predict probabilities
    audio_proba = audio_clf.predict_proba(audio_scaled)
    rav_proba = rav_visual_clf.predict_proba(rav_visual_scaled)
    dc_proba = dc_visual_clf.predict_proba(dc_visual_scaled)
    dc_rav_proba = dc_rav_visual_clf.predict_proba(dc_rav_visual_scaled)

    classes = list(audio_encoder.classes_)
    w_audio, w_rav, w_dc, w_dcrav = (0.33, 0.5, 0.5, 0.8)

    fusion_modes = {
        "ravdess_visual": (w_audio, w_rav, 0.0, 0.0),
        "dcsass_visual": (w_audio, 0.0, w_dc, 0.0),
        "dcsass_ravdess_cat_visual": (w_audio, 0.0, 0.0, w_dcrav),
        "full": (w_audio, w_rav, w_dc, w_dcrav),
    }

    print(f"\n📂 Predicting intent from: {mp4_path}")
    for name, (wa, wrv, wdv, wdrv) in fusion_modes.items():
        fused = wa * audio_proba + wrv * rav_proba + wdv * dc_proba + wdrv * dc_rav_proba
        pred = classes[np.argmax(fused)]
        print(f"{name} ➜ {pred.upper()}")

    print("\n🔍 Prediction Results:")
    print(f"🔊 AUDIO             : {audio_pred}")
    print(f"🎭 RAVDESS           : {rav_visual_pred}")
    print(f"🎬 DCSASS            : {dc_visual_pred}")
    print(f"🧠 DCSASS + RAVDESS  : {dc_rav_visual_pred}")

# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_intent_from_mp4.py path/to/video.mp4")
        exit(1)
    predict_intent_from_mp4(sys.argv[1])
