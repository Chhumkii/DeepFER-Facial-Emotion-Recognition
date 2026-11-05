"""
Real-Time Webcam Emotion Recognition Script
Author: Chumki Nanda (chumkinanda066@gmail.com)
"""

import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLOR_PALETTE = [
    (0, 0, 255),     # Angry: Red
    (0, 128, 0),     # Disgust: Dark Green
    (128, 0, 128),   # Fear: Purple
    (0, 255, 255),   # Happy: Yellow
    (200, 200, 200), # Neutral: Light Gray
    (255, 0, 0),     # Sad: Blue
    (255, 165, 0)    # Surprise: Orange
]

class RealtimeEmotionDetector:
    def __init__(self, model_path="models/custom_cnn.h5", cascade_path=None, smoothing_window=7, alpha=0.3):
        """
        Initializes Haar Cascade face detector, TensorFlow emotion model, and EMA temporal smoothing queue.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded emotion model from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path}: {e}")
            self.model = None

        self.smoothing_window = smoothing_window
        self.alpha = alpha
        self.prob_history = deque(maxlen=smoothing_window)
        self.smoothed_probs = None

    def preprocess_face(self, face_crop):
        """
        Preprocesses cropped face box: converts to 48x48 grayscale, normalizes to [0,1].
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype("float32") / 255.0
        expanded = np.expand_dims(normalized, axis=(0, -1))
        return expanded

    def update_probabilities(self, new_probs):
        """
        Applies Exponential Moving Average (EMA) temporal smoothing over the last 7 frames.
        """
        self.prob_history.append(new_probs)
        if self.smoothed_probs is None:
            self.smoothed_probs = new_probs
        else:
            self.smoothed_probs = self.alpha * new_probs + (1 - self.alpha) * self.smoothed_probs
        return self.smoothed_probs

    def draw_hud(self, frame, x, y, w, h, probs, emotion_label, confidence, fps):
        """
        Draws bounding box, emotion text, live probability bar overlay (HUD), and FPS counter.
        """
        color = COLOR_PALETTE[np.argmax(probs)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{emotion_label}: {confidence*100:.1f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        hud_x = frame.shape[1] - 220
        hud_y = 30
        bar_height = 20
        bar_max_width = 150

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x - 10, hud_y - 20), (frame.shape[1] - 10, hud_y + len(CLASS_NAMES) * 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, (cls_name, p) in enumerate(zip(CLASS_NAMES, probs)):
            y_pos = hud_y + i * 28
            bar_w = int(p * bar_max_width)
            c = COLOR_PALETTE[i]
            cv2.putText(frame, f"{cls_name[:4]}:", (hud_x, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (hud_x + 45, y_pos), (hud_x + 45 + bar_w, y_pos + bar_height), c, -1)
            cv2.putText(frame, f"{int(p*100)}%", (hud_x + 50 + bar_w, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self, camera_id=0):
        """
        Starts live OpenCV webcam stream.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        prev_time = time.time()
        print("Starting OpenCV Real-time Emotion Detector. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if self.model is not None:
                    inp = self.preprocess_face(face_crop)
                    raw_probs = self.model.predict(inp, verbose=0)[0]
                    probs = self.update_probabilities(raw_probs)
                    emotion_idx = np.argmax(probs)
                    emotion_label = CLASS_NAMES[emotion_idx]
                    confidence = probs[emotion_idx]
                else:
                    probs = np.ones(7) / 7.0
                    emotion_label = "Neutral"
                    confidence = 0.5

                self.draw_hud(frame, x, y, w, h, probs, emotion_label, confidence, fps)

            cv2.imshow("DeepFER Real-Time Emotion Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealtimeEmotionDetector()
    detector.run()
