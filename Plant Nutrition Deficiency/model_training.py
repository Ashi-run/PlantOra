import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define dataset paths (using relative paths)
train_dir = os.path.join(os.getcwd(), "dataset", "train_data")
val_dir = os.path.join(os.getcwd(), "dataset", "val_data")

# Verify paths exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError(f"❌ Dataset folders not found!\nTrain: {train_dir}\nVal: {val_dir}")

print("✅ Dataset paths verified!")

# Step 1: Ensure each class folder has at least 9 categories in validation
train_classes = sorted(os.listdir(train_dir))
val_classes = sorted(os.listdir(val_dir))

if train_classes != val_classes:
    raise ValueError(f"❌ Class mismatch! Training Classes: {train_classes}\nValidation Classes: {val_classes}")

print(f"✅ Classes Verified: {train_classes}")

# Step 2: Limit number of images per class to prevent memory issues
def limit_images_per_class(directory, max_files=2000):
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if len(images) > max_files:
                images_to_delete = random.sample(images, len(images) - max_files)
                for img in images_to_delete:
                    os.remove(os.path.join(class_path, img))
                print(f"❗ Reduced {class_folder} class to {max_files} images.")

# Apply dataset reduction
limit_images_per_class(train_dir, max_files=2000)
limit_images_per_class(val_dir, max_files=500)

# Image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Define the classes (from your dataset folder structure)
class_names = ['boron', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese', 'potassium', 'sulphur', 'zinc']

# Initialize the ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load train and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names
)

# Calculate class weights to handle class imbalance
train_labels = np.array(train_generator.classes)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(enumerate(class_weights))

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Add Dropout layer to prevent overfitting
    Dense(len(class_names), activation='softmax')
])

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks (optional)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, checkpoint]
)

# Save the final model
model.save('model.h5')

