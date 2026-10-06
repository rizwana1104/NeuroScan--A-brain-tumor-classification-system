import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Configuration
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
TRAIN_DIR = '../dataset/Training'
TEST_DIR = '../dataset/Testing'
MODEL_SAVE_PATH = 'brain_tumor_model.h5'

def train_model():
    # Data Augmentation
    # EfficientNet expects pixels in [0, 255], so we remove rescale=1./255
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator() # No rescale needed

    # Load Data
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print("Loading validation data...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    # Base Model (Transfer Learning)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Save class indices for inference
    with open('class_indices.txt', 'w') as f:
        f.write(str(train_generator.class_indices))

if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' is empty or does not exist.")
        print("Please place your images in the 'dataset/train' folder before running this script.")
    else:
        train_model()
