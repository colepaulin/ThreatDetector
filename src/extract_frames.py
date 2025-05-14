# import os
# import cv2
# import numpy as np
# from moviepy import *
# from fer import FER
# from tqdm import tqdm

# # -------------------
# # Emotion ID to intent label
# # -------------------
# emotion_to_intent = {
#     "01": "friendly",  # neutral
#     "02": "friendly",  # calm
#     "03": "friendly",  # happy
#     "04": "friendly",  # sad
#     "05": "threat",    # angry
#     "06": "threat",    # fearful
#     "07": "threat",    # disgust
#     "08": "friendly",  # surprised
# }

# # -------------------
# # Helper: Extract FER emotion features
# # -------------------
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
#         return np.mean(frame_features, axis=0)  # 7D feature
#     else:
#         return np.zeros(7)

# # -------------------
# # Main pipeline
# # -------------------
# def extract_visual_features(input_folder = "RAVDESS/Actor_01", output_folder = "features", features_filename = "visual_features.npy", labels_filename = "visual_labels.npy"):
#     all_features = []
#     all_labels = []

#     if not os.path.exists(input_folder):
#         raise FileNotFoundError(f"Actor folder not found: {input_folder}")

#     video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

#     print(f"🎬 Extracting features from {len(video_files)} videos in {input_folder}...")

#     for video_file in tqdm(video_files):
#         video_path = os.path.join(input_folder, video_file)

#         # Extract emotion ID from filename (3rd field)
#         emotion_id = video_file.split("-")[2]

#         # Map to friendly/threat label
#         intent_label = emotion_to_intent.get(emotion_id, None)
#         if intent_label is None:
#             print(f"⚠️ Unknown emotion ID in file: {video_file}")
#             continue

#         # Extract emotion features
#         feature_vector = extract_fer_features(video_path)
#         all_features.append(feature_vector)
#         all_labels.append(intent_label)
    
#     print(f"first 5 label -> features: {all_labels[:5]} map to {all_features[:5]}")

#     os.makedirs(output_folder, exist_ok=True)
#     np.save(os.path.join(output_folder, features_filename), np.array(all_features))
#     np.save(os.path.join(output_folder, labels_filename), np.array(all_labels))

# if __name__ == "__main__":
#     extract_visual_features()

import os
import cv2
import numpy as np
from fer import FER
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# -------------------
# Emotion ID to intent label
# -------------------
emotion_to_intent = {
    "01": "friendly",  # neutral
    "02": "friendly",  # calm
    "03": "friendly",  # happy
    "04": "friendly",  # sad
    "05": "threat",    # angry
    "06": "threat",    # fearful
    "07": "threat",    # disgust
    "08": "friendly",  # surprised
}

# -------------------
# Per-video processor
# -------------------
def process_video(args):
    video_path, intent_label = args
    try:
        detector = FER(mtcnn=True)
        cap = cv2.VideoCapture(video_path)
        frame_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                result = detector.detect_emotions(frame)
                if result:
                    emotions = result[0]["emotions"]
                    vec = [emotions[e] for e in sorted(emotions)]
                    frame_features.append(vec)
            except:
                continue

        cap.release()
        feature = np.mean(frame_features, axis=0) if frame_features else np.zeros(7)
        return (feature, intent_label)
    except Exception as e:
        print(f"❌ Error on {video_path}: {e}")
        return (np.zeros(7), intent_label)

# -------------------
# Main
# -------------------
def extract_visual_features_parallel(base_dir="RAVDESS", output_dir="features", features_filename="visual_features.npy", labels_filename="visual_labels.npy"):
    tasks = []

    for actor_folder in sorted(os.listdir(base_dir)):
        actor_path = os.path.join(base_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for video_file in sorted(os.listdir(actor_path)):
            if not video_file.endswith(".mp4"):
                continue
            video_path = os.path.join(actor_path, video_file)
            emotion_id = video_file.split("-")[2]
            intent = emotion_to_intent.get(emotion_id)
            if intent is not None:
                tasks.append((video_path, intent))

    print(f"🧠 Starting parallel FER extraction on {len(tasks)} videos...")

    with Pool(processes=min(cpu_count(), 6)) as pool:
        results = list(tqdm(pool.imap(process_video, tasks), total=len(tasks)))

    features, labels = zip(*results)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, features_filename), np.array(features))
    np.save(os.path.join(output_dir, labels_filename), np.array(labels))
    print(f"✅ Saved to {output_dir}/{features_filename} and {labels_filename}")


if __name__ == "__main__":
    extract_visual_features_parallel()
