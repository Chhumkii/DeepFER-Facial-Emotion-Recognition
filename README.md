# DeepFER: Facial Emotion Recognition Using Deep Learning

**Author:** Chumki Nanda (chumkinanda066@gmail.com)  
**Repository:** https://github.com/Chhumkii/DeepFER-Facial-Emotion-Recognition

## Project Overview & Background
DeepFER is an end-to-end facial emotion recognition system using deep learning. It classifies human facial expressions into 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. Applications include e-commerce customer satisfaction monitoring, mental health monitoring, and Human-Computer Interaction (HCI).

## Dataset Overview & Preprocessing
- **Image Formats:** 48x48 grayscale and 224x224 RGB images.
- **Class Imbalance:** Mitigated using inverse-frequency class weights during training.
- **Data Augmentation:** Rotation ±15°, scaling/zoom ±15%, and horizontal flip.

## System Architecture
- **Custom DeepFER Residual CNN:** 4-Block custom residual CNN architecture featuring BatchNormalization, LeakyReLU, and Spatial Dropout.
- **Transfer Learning:** MobileNetV2 and ResNet50 backbones.

## Evaluation Metrics & Benchmark Table

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Inference Speed (FPS) |
|---|---|---|---|---|
| Custom DeepFER Residual CNN | 71.4% | 0.698 | 0.712 | 68 FPS |
| MobileNetV2 | 73.8% | 0.724 | — | 52 FPS |
| ResNet50 | 75.1% | — | — | — |

### Per-Class Performance Breakdown & Error Analysis
- Includes per-class precision, recall, and F1 score breakdown across all 7 emotion classes.
- Error analysis highlights primary confusion pairs: Fear vs Surprise and Sad vs Neutral.

## Real-Time Processing & Grad-CAM Explainability
- **Face Detection:** OpenCV Haar Cascade classifier.
- **Temporal Smoothing:** 7-frame Exponential Moving Average (EMA) filter.
- **Visual Overlay & Explainability:** Real-time HUD overlay and Grad-CAM facial landmark visualization.

## Repository Structure & Quickstart
```bash
pip install -r requirements.txt
python -m src.train
python realtime_detector.py
streamlit run app.py


## Pro
