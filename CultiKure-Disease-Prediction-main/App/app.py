import os
import streamlit as st
from PIL import Image
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Define a temporary model (for testing)
class TempModel(nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))

    def forward(self, inp):
        return self.conv1(inp)

# Load pre-trained ResNet50
import torchvision.models as models
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 39)  # Adjust class count

# Load trained weights
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    try:
        img = Image.open(image)
        img = transform(img)
        
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)
        
        return predicted_class.item()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title("PlantOra🌿 Disease Detection")

# Sidebar Navigation
menu = ["Home", "AI Engine", "Market", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

# Home Section
if choice == "Home":
    st.header("Welcome to Plant Disease Detection System")
    sample_image_path = "static/uploads/sample.jpeg"
    
    if os.path.exists(sample_image_path):
        image = Image.open(sample_image_path)
        st.image(sample_image_path, caption="Sample Image", use_column_width=True)
    else:
        st.warning("Sample image not found! Please upload an image.")

# AI Engine Section
elif choice == "AI Engine":
    st.header("Upload an Image for Disease Prediction")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Disease"):
            pred = predict(uploaded_image)
            if pred is not None:
                title = disease_info['disease_name'].iloc[pred]
                description = disease_info['description'].iloc[pred]
                prevent = disease_info['Possible Steps'].iloc[pred]
                image_url = disease_info['image_url'].iloc[pred]
                supplement_name = supplement_info['supplement name'].iloc[pred]
                supplement_image_url = supplement_info['supplement image'].iloc[pred]
                supplement_buy_link = supplement_info['buy link'].iloc[pred]
                
                st.subheader(f"Prediction: {title}")
                st.write(f"**Description:** {description}")
                st.write(f"**Prevention Steps:** {prevent}")
                
                if pd.notna(image_url) and image_url.strip():
                    st.image(image_url, caption=title, use_column_width=True)
                
                st.write(f"**Recommended Supplement:** {supplement_name}")

                if pd.notna(supplement_image_url) and supplement_image_url.strip():
                    st.image(supplement_image_url, caption=supplement_name, use_column_width=True)
                else:
                    st.warning("Supplement image URL not found!")

                st.markdown(f"[Buy Here]({supplement_buy_link})")

# Market Section
elif choice == "Market":
    st.header("Market - Buy Supplements")
    
    if {'supplement image', 'supplement name', 'buy link'}.issubset(supplement_info.columns):
        for i in range(len(supplement_info)):
            supplement_img = supplement_info['supplement image'].iloc[i]
            supplement_name = supplement_info['supplement name'].iloc[i]
            buy_link = supplement_info['buy link'].iloc[i]

            if pd.notna(supplement_img) and supplement_img.strip():
                st.image(supplement_img, caption=supplement_name, use_column_width=True)
            
            st.markdown(f"[Buy Here]({buy_link})")
    else:
        st.warning("Supplement information is missing!")

# Contact Section
elif choice == "Contact":
    st.header("Contact Us")
    st.write("For any inquiries, please reach out to us!")
