import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm.contrib.concurrent import process_map  # ⭐️ Add this
import os
from tqdm import tqdm
from pathlib import Path

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Wav2Vec2 processor and model (make sure to use CUDA if available)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
model.eval()

def extract_wav2vec2_features(wav_path):
    try:
        # Normalize Path
        wav_path = Path(wav_path).as_posix()
        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Tokenize the audio (prepare input for Wav2Vec2)
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract features with Wav2Vec2
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # shape: [1, T, 768]

        # Average pooling over time (T) to get a fixed-size vector
        feature_vector = hidden_states.mean(dim=1).squeeze()
        return feature_vector.cpu().numpy()  # move back to CPU and return as NumPy array

    except Exception as e:
        print(f"❌ Failed on {wav_path}: {e}")
        return None  # Still return something to keep array aligned

def process_audio_files(audio_paths):
    features = []
    for path in tqdm(audio_paths, desc="Extracting Wav2Vec2 features"):
        try:
            vec = extract_wav2vec2_features(path)
            features.append(vec)
        except Exception as e:
            print(f"❌ Failed on {path}: {e}")
            features.append(None)

    return features