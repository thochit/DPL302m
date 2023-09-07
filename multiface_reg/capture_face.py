import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Load the MTCNN face detection model
mtcnn = MTCNN(keep_all=True)

# Create the output directory if it doesn't exist
output_dir = 'multiface_reg/face_crops'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Input person's name
person_name = input("Enter the person's name: ")

# Create a folder for the person's face crops
person_dir = os.path.join(output_dir, person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Open the webcam
cap = cv2.VideoCapture(0)

# Counter to keep track of captured faces
face_count = 0
capture=False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(rgb_frame)

    # Check if 'c' key is pressed to capture the face
    if cv2.waitKey(1) & 0xFF == ord('c'):
        capture = True

    # Check if any faces are detected
    if boxes is not None:
        # Iterate over detected faces
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            # Draw bounding box around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if capture:

                # Crop the face region from the frame
                face_crop = frame[y1:y2, x1:x2]

                # Generate a unique filename for each captured face
                filename = os.path.join(person_dir, f'face_{face_count}.jpg')

                # Save the face crop as an image
                cv2.imwrite(filename, face_crop)
                print(f'face_{face_count}')

                # Increment the face count
                face_count += 1

    # Display the resulting frame
    cv2.imshow('Face Capture', frame)

    # Exit loop on 'q' key press or after capturing 100 faces
    if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= 200:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()