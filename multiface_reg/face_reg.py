import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Create the MTCNN detector and FaceNet embedder
device = 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the SVM classifier from the saved file
svm_model_file = 'multiface_reg/svm_classifier_model.joblib'
svm_classifier = joblib.load(svm_model_file)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform face detection using MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn(img_rgb)
    all_box = mtcnn.detect(img_rgb)[0]
    # print(box)

    if faces is not None and len(faces) > 0:  # Check if faces were detected

        face_embedding = facenet(faces).detach().cpu().numpy()

        # Predict the label using the SVM classifier
        predicted_identities = svm_classifier.predict(face_embedding)
        probabilities = svm_classifier.predict_proba(face_embedding)
        print(probabilities)
        for i, predicted_identity in enumerate(predicted_identities):
            # Draw the predicted label on the frame 
            # box = face[0:4].int().cpu().numpy()
            box = all_box[i].astype(int)

            prediction_prob = np.round(np.max(probabilities[i]),2)

            # Apply a threshold to face recognition
            threshold = 0.9  # Adjust the threshold as needed
            if prediction_prob < threshold:
                predicted_identity = 'Unknown'

            # Define the bottom-left corner of the text
            text_org = (box[0], box[1] - 10)  # Adjust the y-coordinate to place the text just above the bounding box

            # Set the font properties
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 255, 0)  # Green color
            font_thickness = 1

            # Draw the bounding box and predicted label on the frame
            cv2.putText(frame, predicted_identity + ' - ' + str(prediction_prob), text_org, font_face, font_scale, font_color, font_thickness)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
