import json
import os
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from PIL import Image

# Paths
CAPTIONS_FILE = "captions.json"
IMAGES_FOLDER = "images"

# Load model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load captions
with open(CAPTIONS_FILE, "r") as f:
    captions = json.load(f)

# Prepare data for FAISS
def prepare_data(captions):
    embeddings = []
    image_ids = []
    for key, value in captions.items():
        caption = value["caption"]
        image_id = value["filename"]
        embedding = model.encode(caption, convert_to_tensor=True).detach().cpu().numpy()
        embeddings.append(embedding)
        image_ids.append(image_id)
    return np.array(embeddings), image_ids

# Initialize FAISS index
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 (Euclidean) distance
    index.add(embeddings)
    return index

embeddings, image_ids = prepare_data(captions)
faiss_index = create_faiss_index(embeddings)

# Streamlit app
st.title("Image Search Engine")
st.write("Enter a description, and we'll show you the top 3 matching images!")

# User input
query = st.text_input("Enter your query:", "")

if query:
    # Compute query embedding
    query_embedding = model.encode(query, convert_to_tensor=True).detach().cpu().numpy().reshape(1, -1)
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, 3)  # Top 3 results
    st.write(f"Found {len(indices[0])} results!")

    # Display images
    for i, idx in enumerate(indices[0]):
        image_path = os.path.join(IMAGES_FOLDER, image_ids[idx])
        caption = captions[f"image_{idx + 1}"]["caption"]  # Adjusted for 1-based indexing in keys
        st.image(Image.open(image_path), caption=f"Result {i + 1}: {caption}", use_column_width=True)
