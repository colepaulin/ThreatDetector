import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
from tqdm import tqdm

# ----------------------
# Frame Feature Extractor
# ----------------------
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

# ----------------------
# Main Feature Extractor
# ----------------------
def extract_dcsass_features(dataset_dir="DCSASS/DCSASS Dataset", output_dir="features", filename_prefix="dcsass"):
    abuse_dir = os.path.join(dataset_dir, "Abuse")
    stealing_dir = os.path.join(dataset_dir, "Stealing")
    abuse_csv = os.path.join(dataset_dir, "Labels", "Abuse.csv")
    stealing_csv = os.path.join(dataset_dir, "Labels", "Stealing.csv")

    os.makedirs(output_dir, exist_ok=True)
    all_features = []
    all_labels = []

    # Load label CSVs
    abuse_df = pd.read_csv(abuse_csv, header=None, names=["filename", "category", "label"])
    stealing_df = pd.read_csv(stealing_csv, header=None, names=["filename", "category", "label"])

    for df, root_dir in [(abuse_df, abuse_dir), (stealing_df, stealing_dir)]:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {root_dir.split('/')[-1]}"):
            subfolder_name = row["filename"].rsplit("_", 1)[0] + ".mp4" # e.g. Stealing002_x264
            full_video_name = row["filename"] + ".mp4"
            full_path = os.path.join(root_dir, subfolder_name, full_video_name)

            if not os.path.exists(full_path):
                print(f"❌ Missing: {full_path}")
                continue

            label = "threat" if row["label"] == 1 else "friendly"

            try:
                features = extract_action_features(full_path)
                all_features.append(features)
                all_labels.append(label)
            except Exception as e:
                print(f"❌ Error processing {full_video_name}: {e}")

    # Save features
    np.save(os.path.join(output_dir, f"{filename_prefix}_visual_features.npy"), np.array(all_features))
    np.save(os.path.join(output_dir, f"{filename_prefix}_visual_labels.npy"), np.array(all_labels))
    print(f"✅ Saved {len(all_features)} visual samples from DCSASS.")

if __name__ == "__main__":
    extract_dcsass_features()
