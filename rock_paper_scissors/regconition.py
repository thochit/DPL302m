import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('hand_model.h5')

# Set the labels
labels = ['rock', 'paper', 'scissors']

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Set the desired image size
image_size = (64, 64)


# Function to preprocess the image for model input
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox_frame = np.copy(frame)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand region by calculating bounding box
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Apply padding to the bounding box
            padding = 15
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Draw bounding box on separate frame
            cv2.rectangle(bbox_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract hand region with padding
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region image
            preprocessed_image = preprocess_image(hand_roi)

            # Make a prediction using the model
            prediction = model.predict(preprocessed_image)
            label_index = np.argmax(prediction)
            label = labels[label_index]

            # Display the predicted label on the frame
            cv2.putText(bbox_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Recognition', bbox_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
