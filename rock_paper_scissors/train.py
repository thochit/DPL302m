import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import layers

# Set the directory containing the hand captures
data_dir = 'hand_images'

# Set the label names
labels = ['rock', 'paper', 'scissors']

# Set the desired image size
image_size = (64, 64)

# Initialize lists to store the image data and labels
data = []
target = []

# Load and resize the hand captures
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        data.append(image)
        target.append(labels.index(label))

# Convert the data and target lists to NumPy arrays
data = np.array(data)
target = np.array(target)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Create the CNN model
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(labels), activation='softmax')
    ]
)

load_old_model = input('Load old model?[y/n]:')
if load_old_model == 'y':
    model = keras.models.load_model('hand_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save the trained model
model.save('hand_model.h5')
