import kagglehub
import os
import shutil

# Target location (same level as RAVDESS/)
TARGET_DIR = "DCSASS"

# Download latest version
path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

# Move dataset into DCSASS/ if needed
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Copy files from download path to DCSASS/
for root, dirs, files in os.walk(path):
    for file in files:
        src_file = os.path.join(root, file)
        rel_path = os.path.relpath(src_file, path)
        dst_file = os.path.join(TARGET_DIR, rel_path)

        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)

print(f"✅ DCSASS dataset copied to: {os.path.abspath(TARGET_DIR)}")
