from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load fine-tuned model
model = load_model('model.h5')  # Ensure model.h5 exists

# Define class names
class_names = ['boron', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese', 'potassium', 'sulphur', 'zinc']

def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"\u274c Image not found: {image_path}")

    img = image.load_img(image_path, target_size=(128, 128))  # Ensure it matches training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize as done in training

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return class_names[predicted_class]  # Return only the predicted class

# Example usage

# Define relative path
image_path = os.path.join("dataset", "train_data", "healthy", "h_338.jpg")  

# Predict image
predicted_label = predict_image(image_path)
print(f"Predicted class: {predicted_label}")
