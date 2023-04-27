from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data = []
labels = []
ParentDir = os.path.abspath('Cyrillic')
ChildDirs = os.listdir(ParentDir)

#Load the dataset using an image processing library like OpenCV or PIL
for folder in ChildDirs:
    label = folder
    child_folder_path = os.path.join(ParentDir, folder)

    for img_name in os.listdir(child_folder_path):
        img_path = os.path.join(child_folder_path, img_name)

        img = Image.open(img_path)
        
        if img is not None:
            img = img.convert('L')
            data.append(img)
            labels.append(label)
        else:
            print(f"Failed to read image: {img_path}")
#Preprocess the images by resizing them to a smaller size, normalizing the pixel values, and converting the labels to one-hot encoding:
# Resize images to 28x28
data = [np.array(img.resize((28, 28), resample=Image.BICUBIC)) for img in data]


# Normalize pixel values to [0, 1]
data = np.array(data) / 255.0

# Convert labels to one-hot encoding
unique_labels = list(set(labels))
label_dict = {label: i for i, label in enumerate(unique_labels)}
labels = [label_dict[label] for label in labels]
labels = np.eye(len(unique_labels))[labels]

#Split the dataset into training and validation sets:
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

#CNNs are the most commonly used architecture for image classification tasks:
'''
This creates a simple CNN with two convolutional layers, two max pooling layers, a fully connected layer with 128 units, a dropout layer for regularization, 
and a final output layer with softmax activation.
'''
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

#Compile the model with an appropriate loss function, optimizer, and metrics:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model on the training set using the fit method:
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

#Evaluate the model on the validation set using the evaluate method:
loss, accuracy = model.evaluate(val_data, val_labels)
print(f"Validation loss: {loss:.4f}, accuracy: {accuracy:.4f}")


# Save the weights to a file
model.save_weights('cyrillic_weights.h5')

# Load the weights from the file
#model.load_weights('cyrillic_weights.h5')