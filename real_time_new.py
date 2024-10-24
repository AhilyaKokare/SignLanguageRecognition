# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('sign_language_detection_model.h5')

# # Define the list of categories (A-Z)
# categories = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# # Function to preprocess each frame
# def preprocess_image(frame):
#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Resize to 64x64 (same size used during training)
#     resized_frame = cv2.resize(gray_frame, (64, 64))
#     # Normalize the pixel values (0-1 range)
#     normalized_frame = resized_frame / 255.0
#     # Reshape to (1, 64, 64, 1) for prediction
#     reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 1))
#     return reshaped_frame

# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)

# # Loop for real-time detection
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Preprocess the frame
#     processed_frame = preprocess_image(frame)

#     # Predict the alphabet using the model
#     predictions = model.predict(processed_frame)
#     predicted_class = np.argmax(predictions)
#     predicted_label = categories[predicted_class]

#     # Display the predicted alphabet on the frame
#     cv2.putText(frame, f'Predicted Alphabet: {predicted_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame with the prediction
#     cv2.imshow('Sign Language Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()














# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('sign_language_detection_model.h5')
# # model = load_model('D:/LY_CSE/DL/Mini_Project/Sign_Language_Detection/sign_language_detection_model.h5')


# # Define the list of categories (A-Z)
# categories = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# # Function to preprocess each frame
# def preprocess_image(frame):
#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Resize to 64x64 (same size used during training)
#     resized_frame = cv2.resize(gray_frame, (64, 64))
#     # Normalize the pixel values (0-1 range)
#     normalized_frame = resized_frame / 255.0
#     # Reshape to (1, 64, 64, 1) for prediction
#     reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 1))
#     return reshaped_frame

# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)

# # Loop for real-time detection
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Preprocess the frame
#     processed_frame = preprocess_image(frame)

#     # Predict the alphabet using the model
#     predictions = model.predict(processed_frame)
#     predicted_class = np.argmax(predictions)
#     confidence = np.max(predictions)

#     # Confidence threshold
#     confidence_threshold = 0.5
#     if confidence > confidence_threshold:
#         predicted_label = categories[predicted_class]
#     else:
#         predicted_label = "Unknown"

#     # Display the predicted alphabet on the frame
#     cv2.putText(frame, f'Predicted Alphabet: {predicted_label} (Confidence: {confidence:.2f})', 
#                 (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame with the prediction
#     cv2.imshow('Sign Language Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()














import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model using an absolute path
# model_path = r'D:/LY_CSE/DL/Mini_Project/Sign_Language_Detection/sign_language_detection_model.h5'
model_path = r'D:/PROGRAM/ML/Sign_Language_Detection/sign_language_detection_model.h5'

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Define the list of categories (A-Z)
categories = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Function to preprocess each frame
def preprocess_image(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 64x64 (same size used during training)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    # Normalize the pixel values (0-1 range)
    normalized_frame = resized_frame / 255.0
    # Reshape to (1, 64, 64, 1) for prediction
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 1))
    return reshaped_frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Loop for real-time detection
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Predict the alphabet using the model
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Confidence threshold
    confidence_threshold = 0.5
    if confidence > confidence_threshold:
        predicted_label = categories[predicted_class]
    else:
        predicted_label = "Unknown"

    # Display the predicted alphabet on the frame
    cv2.putText(frame, f'Predicted Alphabet: {predicted_label} (Confidence: {confidence:.2f})', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the prediction
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
