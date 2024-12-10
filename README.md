## Project Overview
This project implements an AI-powered yoga pose detection and correction system. It identifies yoga poses from images, evaluates the accuracy of the detected poses, and provides real-time feedback for correction. Using **Mediapipe Pose**, **TensorFlow**, and machine learning techniques, the system achieves high efficiency in pose classification and correction.



## Features
1. **Pose Landmark Extraction**: 
    Extracts pose landmarks from images using Mediapipe and saves extracted landmarks as JSON files for further processing.

2. **Data Preprocessing**:
   - Normalizes landmarks for consistency.
   - Splits data into training and testing sets.

3. **Model Training**:
   - Builds and trains a neural network for pose classification.
   - Includes dropout and batch normalization layers to enhance accuracy and reduce overfitting.

4. **Evaluation**:
   - Evaluates the trained model using classification metrics such as accuracy, confusion matrix, and classification report.

5. **Real-Time Pose Correction**:
   - Provides real-time feedback on detected poses using webcam input.
   - Displays the predicted pose name and confidence score.



## Requirements
### Software and Libraries
- Python 3.8+
- TensorFlow 2.x
- Mediapipe
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install required libraries using:
```bash
pip install tensorflow mediapipe opencv-python-headless numpy matplotlib seaborn scikit-learn
```

### Hardware Requirements
- A system with a working webcam (for real-time pose correction).
- GPU support (optional but recommended for faster training).



## Dataset : YOGA-82(prexisting)
Source: Kaggle
### Structure
- **Annotated Folder**:
  - Contains subfolders for each yoga pose.
  - Each subfolder includes images of that pose.
- **Landmarks Folder**:
  - Populated with extracted landmark JSON files after running the pipeline.



## How It Works
### Workflow
1. **Landmark Extraction**:
   - Extracts 3D pose landmarks from images in the `Annotated` folder.
   - Saves these landmarks in the `Landmarks` folder as JSON files.

2. **Data Preprocessing**:
   - Loads landmarks and prepares them for training.
   - Normalizes landmarks to improve consistency.

3. **Model Training**:
   - Builds a fully connected neural network.
   - Trains the model on preprocessed data.

4. **Evaluation**:
   - Evaluates model performance using a test set.
   - Generates a classification report and confusion matrix.

5. **Real-Time Pose Correction**:
   - Uses a webcam to capture live video.
   - Detects pose landmarks in real time and predicts the yoga pose.
   - Provides feedback with pose name and confidence score.



## Code Structure
- **`YogaPosePipeline` Class**:
  - Handles all major functionality: landmark extraction, preprocessing, training, evaluation, and real-time correction.
- **`main()` Function**:
  - Drives the pipeline execution.



## How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/yoga-pose-detection.git
cd yoga-pose-detection
```

### 2. Set Up the Dataset
- Place the yoga pose images in the `Annotated` folder.
- Ensure each subfolder is named after its respective pose (e.g., `Tree`, `Warrior`).

### 3. Run the Pipeline
```bash
python YogaPoseDetectionandCorrection.py
```

---

## Outputs
1. **Training**:
   - Trained model is saved as `yoga_pose_model.h5`.
2. **Evaluation**:
   - Overall accuracy score.
   - Classification report with precision, recall, and F1 scores.
   - Confusion matrix visualization.
3. **Real-Time Correction**:
   - Predicted pose name and confidence displayed on webcam feed.




## Key Code Functions
### Landmark Extraction
```python
def extract_landmarks(self):
    # Extracts pose landmarks from images and saves them as JSON files.
```

### Model Training
```python
def build_model(self):
    # Builds a neural network for pose classification.
```

### Real-Time Correction
```python
def real_time_correction(self):
    # Uses a webcam to detect and correct yoga poses in real time.
```


## Improvements and Future Work
1. **Expand Dataset**: Add more images for better generalization.
2. **Transfer Learning**: Incorporate pre-trained models for feature extraction.
3. **Advanced Feedback**: Add suggestions for correcting alignment or posture.

