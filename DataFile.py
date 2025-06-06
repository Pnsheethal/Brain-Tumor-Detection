# Install TensorFlow if needed (usually pre-installed in Colab)
!pip install -q tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt

# Parameters
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

# 1. Prepare data generator assuming folder structure:
# dataset/
#    NoTumor/
#    BenignTumor/
#    MalignantTumor/

print("⚠️ Upload your dataset ZIP with three folders: NoTumor, BenignTumor, MalignantTumor")
uploaded_dataset = files.upload()

import zipfile
import os

for zip_filename in uploaded_dataset.keys():
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('dataset')

# Data augmentation & preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # multi-class
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# 2. Define CNN model for 3 classes
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 4. Save model
model.save('brain_tumor_model_3class.keras')
print("✅ Model saved as brain_tumor_model_3class.keras")

# 5. Upload an MRI image to predict tumor class
print("📤 Upload an MRI image (jpg or png) for tumor detection:")
uploaded_images = files.upload()

for img_name in uploaded_images.keys():
    # Load and preprocess image
    img = load_img(img_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    # Predict
    prediction = model.predict(img_array)[0]  # 3-class probs
    class_idx = np.argmax(prediction)

    classes = ['No Tumor', 'Benign Tumor', 'Malignant Tumor']
    confidence = prediction[class_idx]

    print(f"Prediction: {classes[class_idx]} with confidence {confidence:.2f}")

    # Display uplaaoaded image
    plt.imshow(img)
    plt.title(f"Prediction: {classes[class_idx]} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
