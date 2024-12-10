import os
import cv2
import json
from PIL import Image
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Yoga Pose Pipeline Class
class YogaPosePipeline:
    def __init__(self, dataset_path, model_path=None):
        self.dataset_path = dataset_path
        self.landmarks_path = os.path.join(dataset_path, "Landmarks")
        self.model_path = model_path or 'yoga_pose_model.h5'
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.label_encoder = LabelEncoder()
        self.model = None

    def normalize_landmarks(self, landmarks):
        landmarks = np.array(landmarks)
        mid_hip = (landmarks[23] + landmarks[24]) / 2
        normalized_landmarks = landmarks - mid_hip
        return normalized_landmarks.flatten()

    def preprocess_data(self):
        print("Preprocessing data...")
        self.X = []
        self.y = []
        for category in os.listdir(self.landmarks_path):
            category_path = os.path.join(self.landmarks_path, category)
            for file in os.listdir(category_path):
                with open(os.path.join(category_path, file), 'r') as f:
                    data = json.load(f)
                    landmarks_flat = [coord for landmark in data['landmarks'] for coord in landmark]
                    self.X.append(landmarks_flat)
                    self.y.append(category)

        self.X = np.array(self.X)
        self.y = self.label_encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def build_model(self, input_shape, num_classes):
        print("Building the model...")
        self.model = Sequential([
            Input(shape=(input_shape,)),  # Use the Input layer here to define the shape
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, epochs=50, batch_size=32):
        print("Training the model...")
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def evaluate_model(self):
        print("Evaluating the model...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        # Ensure label alignment
        labels = range(len(self.label_encoder.classes_))

        print("\nClassification Report:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.label_encoder.classes_,
            labels=labels,
            zero_division=0  # Suppress undefined metric warnings
        ))

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")

        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def real_time_correction(self, labels):
        print("Starting real-time pose correction...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                normalized_landmarks = self.normalize_landmarks(landmarks)

                prediction = self.model.predict(normalized_landmarks.reshape(1, -1))
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                cv2.putText(frame, f"Pose: {labels[predicted_class]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Yoga Pose Correction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Main Function
def main():
    DATASET_PATH = r"C:\Users\Rimjhim\Desktop\Yoga Pose Detection and Correction"
    pipeline = YogaPosePipeline(DATASET_PATH)
    
    pipeline.preprocess_data()
    input_shape = pipeline.X_train.shape[1]
    num_classes = pipeline.y_train.shape[1]
    
    pipeline.build_model(input_shape, num_classes)
    pipeline.train_model()
    pipeline.evaluate_model()
    
    labels = pipeline.label_encoder.classes_
    pipeline.real_time_correction(labels)


if __name__ == "__main__":
    main()
