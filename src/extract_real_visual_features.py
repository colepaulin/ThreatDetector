import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

# Torch + ResNet
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# -------------------
# Load labels from CSV
# -------------------
def load_real_labels(label_csv="test_inputs/labels.csv", input_folder="test_inputs"):
    real_dataset = []
    try:
        with open(label_csv, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                filename = row["file"]
                label = row["label"]
                video_path = os.path.join(input_folder, filename)
                if os.path.exists(video_path) and filename.endswith(".mp4"):
                    real_dataset.append((video_path, label))
    except Exception as e:
        print(f"❌ Failed to read labels: {e}")
    return real_dataset

# -------------------
# (Commented out) Helper: Extract FER emotion features
# -------------------
# from fer import FER
# def extract_fer_features(video_path):
#     detector = FER(mtcnn=True)
#     cap = cv2.VideoCapture(video_path)
#     frame_features = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         try:
#             result = detector.detect_emotions(frame)
#             if result:
#                 emotions = result[0]["emotions"]
#                 vec = [emotions[emotion] for emotion in sorted(emotions)]
#                 frame_features.append(vec)
#         except Exception as e:
#             print(f"❌ Error on {video_path}: {e}")
#             continue
#     cap.release()
#     if frame_features:
#         return np.mean(frame_features, axis=0)
#     else:
#         return np.zeros(7)

# -------------------
# ResNet feature extractor (matches DCSASS)
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = resnet50(pretrained=True).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_resnet_features(video_path):
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
                    feat = resnet(img).squeeze().cpu().numpy()
                    frame_features.append(feat)
            except Exception as e:
                print(f"❌ ResNet error on {video_path}: {e}")
        idx += 1

    cap.release()
    return np.mean(frame_features, axis=0) if frame_features else np.zeros(2048)

# -------------------
# Main pipeline
# -------------------
def extract_real_visual_features_resnet(
    input_folder="test_inputs",
    label_csv="test_inputs/labels.csv",
    output_folder="features",
    features_filename="real_dcsass_visual_features.npy",
    labels_filename="real_dcsass_visual_labels.npy"
):
    all_features = []
    all_labels = []

    real_dataset = load_real_labels(label_csv, input_folder)
    print(f"🎬 Extracting ResNet features from {len(real_dataset)} real videos...")

    for video_path, intent_label in tqdm(real_dataset):
        feature_vector = extract_resnet_features(video_path)
        all_features.append(feature_vector)
        all_labels.append(intent_label)

    print(f"✅ Example mappings: {all_labels[:3]} → [feature vectors]")

    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, features_filename), np.array(all_features))
    np.save(os.path.join(output_folder, labels_filename), np.array(all_labels))

if __name__ == "__main__":
    extract_real_visual_features_resnet()
