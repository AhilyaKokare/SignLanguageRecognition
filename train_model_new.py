# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical

# # Define paths
# data_dir = "SignImage48x48"  # Replace with your dataset path
# categories = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# # Prepare dataset
# def load_data(data_dir, categories):
#     data = []
#     labels = []
    
#     for category in categories:
#         path = os.path.join(data_dir, category)
#         class_num = categories.index(category)
        
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 img_resized = cv2.resize(img_array, (64, 64))  # Resize images to 64x64 pixels
#                 data.append(img_resized)
#                 labels.append(class_num)
#             except Exception as e:
#                 pass

#     return np.array(data), np.array(labels)

# # Load dataset
# data, labels = load_data(data_dir, categories)

# # Normalize data
# data = data / 255.0

# # Reshape data for the CNN model (64x64 pixels, 1 channel for grayscale)
# data = data.reshape(-1, 64, 64, 1)

# # Convert labels to one-hot encoding
# labels = to_categorical(labels, num_classes=len(categories))

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Build the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
#     MaxPooling2D(2, 2),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
    
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
    
#     Dense(len(categories), activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# # Save the model
# model.save("sign_language_detection_model.h5")

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {accuracy*100:.2f}%")

# # Plot training and validation accuracy
# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()












import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set paths
directory = 'SignImage48x48/'

# Load dataset
data = []
labels = []
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

for label in classes:
    path = os.path.join(directory, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        data.append(img)
        labels.append(classes.index(label))

# Convert to NumPy arrays and normalize
data = np.array(data) / 255.0
data = np.expand_dims(data, axis=-1)  # Add channel dimension for grayscale images
labels = to_categorical(np.array(labels), num_classes=27)  # Updated for 27 classes (including 'blank')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(27, activation='softmax')  # Updated for 27 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save('sign_language_model1.keras')
