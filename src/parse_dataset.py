import os
import csv

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_ravdess_dataset(base_path):
    data = []
    for actor_folder in sorted(os.listdir(base_path)):
        actor_path = os.path.join(base_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for filename in sorted(os.listdir(actor_path)):
            if filename.endswith(".mp4"):
                full_path = os.path.join(actor_path, filename)
                emotion_code = filename.split("-")[2]
                emotion = emotion_map.get(emotion_code, "unknown")
                data.append((full_path, emotion))
    return data

def parse_real_dataset(base_path="test_inputs", label_file="test_inputs/labels.csv"):
    """
    Parses real-world video dataset based on a labels.csv file.

    Args:
        base_path (str): Path to folder containing .mp4 files.
        label_file (str): CSV file containing filename,label pairs.

    Returns:
        List of (full_path, label) tuples.
    """
    data = []
    try:
        with open(label_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row["file"]
                label = row["label"]
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path) and filename.endswith(".mp4"):
                    data.append((full_path, label))
    except Exception as e:
        print(f"❌ Error parsing real dataset labels: {e}")
    return data

