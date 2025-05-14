import os
from moviepy.editor import VideoFileClip

def extract_audio_from_video(mp4_path, wav_path):
    try:
        video = VideoFileClip(mp4_path)
        audio = video.audio
        audio.write_audiofile(wav_path, codec='pcm_s16le', logger=None)
        print(f"✅ Extracted: {wav_path}")
    except Exception as e:
        print(f"❌ Failed: {mp4_path} — {e}")

def batch_extract_audio(data, input_root="RAVDESS", output_root="RAVDESS_audio"):
    for mp4_path, _ in data:
        # Rebuild relative path
        rel_path = os.path.relpath(mp4_path, input_root)
        wav_path = os.path.join(output_root, rel_path).replace(".mp4", ".wav")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        # Skip if already done
        if os.path.exists(wav_path):
            print(f"⏩ Skipping (already exists): {wav_path}")
            continue

        extract_audio_from_video(mp4_path, wav_path)
