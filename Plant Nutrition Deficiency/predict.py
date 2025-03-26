import cv2
import numpy as np
from tensorflow.keras.models import load_model  # ✅ Ensure this import is here

# Load the model
model = load_model('model.h5')  # Ensure 'model.h5' exists in the same directory

# Class labels
class_labels = ['boron', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese', 'potassium', 'sulphur', 'zinc']

def predict_deficiency(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Ensure this matches the model's input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return class_labels[class_index], prediction[0][class_index]
