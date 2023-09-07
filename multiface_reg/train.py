import numpy as np
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Create the MTCNN detector and FaceNet embedder
device = 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the labeled dataset and extract face embeddings and labels
def load_labeled_dataset(data_folder):
    face_embeddings = []
    labels = []
    # print(os.listdir(data_folder))
    for label in os.listdir(data_folder):
        print(label)
        label_folder = os.path.join(data_folder, label)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            face_embedding = extract_face_embedding(image_path)
            if face_embedding is not None:  # Check if a face was detected
                face_embeddings.append(face_embedding)
                labels.append(label)
    return np.array(face_embeddings), np.array(labels)

# Extract the face embedding using FaceNet
def extract_face_embedding(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        faces = mtcnn(img_rgb)
    except:
        return None
    if faces is not None and len(faces) > 0:  # Check if faces were detected
        face_embedding = facenet(faces).detach().numpy()
        face_embedding = np.array(face_embedding).reshape(-1)
        return face_embedding
    else:
        return None

# Train the SVM classifier on the face embeddings
def train_svm_classifier(X, y):
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(X, y)
    return svm_classifier

if __name__ == '__main__':
    # Assuming you have a folder containing face images with corresponding labels
    data_folder = 'multiface_reg/face_crops'

    # Load the labeled dataset and extract face embeddings and labels
    X, y = load_labeled_dataset(data_folder)

    # Remove samples with None embeddings
    mask = [embedding is not None for embedding in X]
    X = X[mask]
    y = y[mask]

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(y_encoded)

    # Train the SVM classifier on the training data
    svm_classifier = train_svm_classifier(X, y)

    # Save the SVM classifier to a file
    svm_model_file = 'svm_classifier_model.joblib'
    joblib.dump(svm_classifier, svm_model_file)
