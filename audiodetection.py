import os
import librosa
import numpy as np
from sklearn.svm import SVC
import joblib
from moviepy.editor import VideoFileClip
# Step 1: Feature Extraction
def extract_features(audio_file, num_mfcc=13):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Load data from 'truth' and 'lie' folders
def load_data(data_dir):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            label_value = 1 if label == 'truth_audio' else 0  # Assign 1 for 'truth_audio' and 0 otherwise
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(label_dir, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label_value)
    return np.array(X), np.array(y)

# Step 2: Train Model
def train_model(X, y):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X, y)
    return svm_classifier

# Step 3: Save Model
def save_model(svm_classifier, model_path):
    joblib.dump(svm_classifier, model_path)

# Step 4: Deployment (Prediction)
def predict_deception(audio_file, svm_classifier):
    features = extract_features(audio_file)
    prediction = svm_classifier.predict([features])[0]
    return prediction

# Prediction function
def deception_prediction(audio_file_path, filename2, model_path):
    # Write audio file from video clip
    video_clip = VideoFileClip(filename2)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file_path, codec='pcm_s16le', bitrate='256k')

    # Load pre-trained SVM model
    svm_classifier = joblib.load(model_path)

    # Predict deception for the specified audio file
    prediction = predict_deception(audio_file_path, svm_classifier)
    
    return prediction
