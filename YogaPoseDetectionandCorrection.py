import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


class YogaPosePipeline:
    def __init__(self, annotated_path, landmarks_path, model_path=None):
        self.annotated_path = annotated_path
        self.landmarks_path = landmarks_path
        self.model_path = model_path or 'yoga_pose_model.h5'
        self.label_encoder = LabelEncoder()
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.X = []
        self.y = []
        self.model = None

    def normalize_landmarks(self, landmarks):
        """
        Normalize the landmarks using a bounding box approach for better consistency.
        """
        landmarks = np.array(landmarks)
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]

        # Calculate bounding box
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Normalize to [0, 1]
        normalized_landmarks = (landmarks - [x_min, y_min, 0, 0]) / [(x_max - x_min), (y_max - y_min), 1, 1]
        return normalized_landmarks.flatten()

    def extract_landmarks(self):
        """
        Extract landmarks from images and save them as JSON files in the Landmarks directory.
        """
        if not os.path.exists(self.landmarks_path):
            os.makedirs(self.landmarks_path)

        print(f"Extracting landmarks from images in: {self.annotated_path}")
        for category in os.listdir(self.annotated_path):
            category_path = os.path.join(self.annotated_path, category)
            landmarks_category_path = os.path.join(self.landmarks_path, category)

            if not os.path.isdir(category_path):
                continue

            os.makedirs(landmarks_category_path, exist_ok=True)
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Skipping corrupted file: {file_path}")
                    continue

                # Resize for faster processing
                image = cv2.resize(image, (96, 96))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = [
                        [lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark
                    ]
                    output_file = os.path.join(landmarks_category_path, f"{os.path.splitext(file)[0]}.json")
                    with open(output_file, 'w') as f:
                        json.dump({"landmarks": landmarks, "label": category}, f)

        print("Landmark extraction completed.")

    def preprocess_data(self):
        """
        Load landmarks from JSON files and prepare the dataset.
        """
        print(f"Loading landmarks from: {self.landmarks_path}")
        for category in os.listdir(self.landmarks_path):
            category_path = os.path.join(self.landmarks_path, category)
            if not os.path.isdir(category_path):
                continue

            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    landmarks_flat = [coord for landmark in data['landmarks'] for coord in landmark]
                    self.X.append(landmarks_flat)
                    self.y.append(data['label'])

        self.X = np.array(self.X)
        self.y = self.label_encoder.fit_transform(self.y)

        if len(self.X) == 0:
            raise ValueError("No landmarks found. Please check your dataset.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(f"Data split: {len(self.X_train)} train samples, {len(self.X_test)} test samples.")

    def build_model(self):
        print("Building the model...")
        input_shape = (self.X_train.shape[1],)
        self.model = Sequential([
            Input(shape=input_shape),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model built successfully.")

    def train_model(self, epochs=15, batch_size=32):
        print("Training the model...")
        self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        print("Training completed.")

    def evaluate_model(self):
        print("Evaluating the model...")
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Align labels to classes present in y_test
        unique_labels = np.unique(self.y_test)

        print("Classification Report:")
        print(classification_report(
            self.y_test, y_pred_classes,
            target_names=self.label_encoder.inverse_transform(unique_labels),
            labels=unique_labels,
            zero_division=0
        ))

        accuracy = accuracy_score(self.y_test, y_pred_classes)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")

        cm = confusion_matrix(self.y_test, y_pred_classes, labels=unique_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=self.label_encoder.inverse_transform(unique_labels),
                    yticklabels=self.label_encoder.inverse_transform(unique_labels))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def real_time_correction(self):
        print("Starting real-time pose correction...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark
                ]
                normalized_landmarks = np.array(landmarks).flatten()

                prediction = self.model.predict(normalized_landmarks.reshape(1, -1))
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                cv2.putText(frame, f"Pose: {self.label_encoder.classes_[predicted_class]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Yoga Pose Correction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    annotated_path = r"C:\Users\Rimjhim\Desktop\Yoga Pose Detection and Correction\Annotated"
    landmarks_path = r"C:\Users\Rimjhim\Desktop\Yoga Pose Detection and Correction\Landmarks"

    pipeline = YogaPosePipeline(annotated_path, landmarks_path)
    pipeline.extract_landmarks()
    pipeline.preprocess_data()
    pipeline.build_model()
    pipeline.train_model(epochs=15)
    pipeline.evaluate_model()
    pipeline.real_time_correction()


if __name__ == "__main__":
    main()
