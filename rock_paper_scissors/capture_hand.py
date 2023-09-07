import cv2
import mediapipe as mp
import os

# Set the output directory
output_dir = 'hand_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the number of images to capture for each label
num_images_per_label = 400

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Label names and counters
labels = ['rock', 'paper', 'scissors']
label_counters = {label: 0 for label in labels}

# Variable to track the currently active label
active_label = None


# Create a function to save the image with the label
def save_image_with_label(frame, label):
    label_counters[label] += 1
    folder = os.path.join(output_dir, label)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f'{folder}/{label}_{label_counters[label]}.jpg'
    cv2.imwrite(filename, frame)
    print(f'Saved image: {filename}')


cap = cv2.VideoCapture(0)

capture = False

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Save hand region with label if capture is enabled
            if capture and active_label:
                if label_counters[active_label] < num_images_per_label:
                    save_image_with_label(hand_roi, active_label)
                else:
                    print(f'Reached the maximum number of images for "{active_label}" label: {num_images_per_label}')

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))

    cv2.imshow('Hand Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        active_label = 'rock'
        print(f'Active label: {active_label}')
    elif key == ord('p'):
        active_label = 'paper'
        print(f'Active label: {active_label}')
    elif key == ord('s'):
        active_label = 'scissors'
        print(f'Active label: {active_label}')
    elif key == ord('c'):
        capture = True

cap.release()
cv2.destroyAllWindows()
