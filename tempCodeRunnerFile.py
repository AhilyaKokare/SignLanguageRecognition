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














import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('sign_language_model1.keras')

# Define the classes (A-Z and blank)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Draw rectangle for Region of Interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    roi = frame[40:300, 0:300]
    
    # Preprocess the ROI
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))  # Add batch dimension and channel dimension
    
    # Predict the letter
    prediction = model.predict(roi_reshaped)
    class_index = np.argmax(prediction)
    predicted_letter = classes[class_index]
    
    # Display the predicted letter on the frame
    cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
