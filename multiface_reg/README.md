# Face Recognition Project

This is a simple face recognition project that allows you to capture faces, train a recognition model, and perform face recognition.

## Installation

Before running the project, make sure you have the following Python packages installed:

- numpy
- opencv-python (cv2)
- scikit-learn (sklearn)
- facenet-pytorch
- torch
- joblib

You can install these packages using pip:

```shell
pip install numpy opencv-python scikit-learn facenet-pytorch torch joblib 
```
## Usage
### Capture Faces (capture_face.py)

To capture faces and store them for later recognition, follow these steps:

Open your terminal or command prompt.

Navigate to the directory containing the project files.

Run the following command:

```bash
python capture_face.py
```
Follow the on-screen instructions to capture faces. The captured face images will be stored in a folder named face_crops.

### Train the Model (train.py)
To train the face recognition model using the captured face images, perform the following:

Open your terminal or command prompt.

Navigate to the directory containing the project files.

Run the following command:

```bash
python train.py
```

The training process will start, and the trained model will be saved as svm_classifier_model.joblib using joblib.

### Face Recognition (face_reg.py)

To recognize faces after capturing and training, follow these steps:

Open your terminal or command prompt.

Navigate to the directory containing the project files.

Run the following command:

```bash
python face_reg.py
```
Follow the on-screen instructions to perform face recognition.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or further information, please reach out to us at thochit2762003@gmail.com.

---

