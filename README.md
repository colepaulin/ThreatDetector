# ThreatDetector

A privacy-preserving multimodal intent classification system for residential doorbell footage. This project uses audio and visual data to classify interactions as either **friendly** or **threatening** using late fusion of multiple classifiers.

---

## File Structure

### `features/`
Contains the extracted NumPy feature vectors for all classifiers:
- Audio (Wav2Vec2)
- Visual (FER + ResNet-based)
- Combined modalities

### `models/`
Saved `MLPClassifier` models after training and fine-tuning:
- Audio-only
- Visual (FER and ResNet)
- Fused classifier
- Fine-tuned classifiers on real-world videos

### `src/`
All pipeline scripts are organized by purpose:

- **Training**
  - `train_*`: Train models on extracted features
- **Fine-tuning**
  - `fine_tune_*`: Fine-tune models using real-world labeled video data
- **Evaluation**
  - `evaluate_*`: Evaluate the performance of trained models
- **Feature Extraction**
  - `extract_*`: Extract visual and audio features from videos
- **Prediction**
  - `predict_intent_from_mp4.py`: Main entry point to run predictions on test `.mp4` files  
    **Usage:**  
    ```bash
    python src/predict_intent_from_mp4.py test_inputs/<video>.mp4
    ```
- **Fusion**
  - `weighted_vote_fusion.py`: Performs final classification using late fusion.  
    Adjust weights in the file to emphasize different modalities.

---

## Test Data

### `test_inputs/`
Contains the 23 custom real-world `.mp4` video inputs.  
Also includes a `labels.csv` file with ground truth labels for evaluation.

### `test_inputs_audio/`
Contains the corresponding `.wav` files extracted from `test_inputs/`.

---

## Setup

Install dependencies inside a virtual environment:

```bash
pip install -r requirements.txt
