import streamlit as st
import cv2
import numpy as np
import os
import torch
from torchvision import transforms, models
from sklearn.metrics.pairwise import cosine_similarity

# Load ResNet-50 Model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model

model = load_model()

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load manually extracted embeddings
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    image_paths = np.load("image_paths.npy")
    person_ids = np.load("person_ids.npy")
    return embeddings, image_paths, person_ids

embeddings, image_paths, person_ids = load_data()

# Extract features for the uploaded image
def extract_query_embedding(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        embedding = model(img).squeeze().cpu().numpy()

    return embedding

# Search for the most similar person and retrieve all images
def search_similar_person(query_embedding, embeddings, image_paths, person_ids, top_k=3):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    best_match_idx = similarities.argsort()[::-1][0]  # Get most similar image index
    best_person_id = person_ids[best_match_idx]

    # Retrieve ALL images belonging to this person
    matched_images = [img for img, pid in zip(image_paths, person_ids) if pid == best_person_id]

    # Limit to `top_k` images if needed
    matched_images = matched_images[:top_k]  

    print(f"✅ Found {len(matched_images)} images for Person ID: {best_person_id}")  # Debugging

    return best_person_id, matched_images


# Streamlit UI
st.title("🔍 Person Search & Retrieval")
st.write("Upload an image to find all images of the same person.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract embedding
    st.write("🔄 Extracting features...")
    query_embedding = extract_query_embedding(image)

    # Find similar person
    st.write("🔍 Searching for similar person...")
    best_person_id, matched_images = search_similar_person(query_embedding, embeddings, image_paths, person_ids)

    # Display results
    st.subheader(f"🔎 Found Person: {best_person_id}")

    # Debugging
    st.write(f"Total retrieved images: {len(matched_images)}")

    # Show retrieved images
    cols = st.columns(min(len(matched_images), 3))  # Dynamically adjust columns

    for i, img_path in enumerate(matched_images):
        cols[i].image(img_path, caption=os.path.basename(img_path), use_column_width=True)


st.write("This system retrieves **all images** belonging to the **same person**.")