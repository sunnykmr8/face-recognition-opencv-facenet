import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
import matplotlib.pyplot as plt

# Initialize MTCNN and FaceNet models
detector = MTCNN()
embedder = FaceNet()

# Function to process image and perform face detection + recognition
def process_image(image):
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV format
    
    # Detect faces using MTCNN
    results = detector.detect_faces(rgb_image)
    
    if len(results) == 0:
        st.write("No faces detected.")
        return
    
    # Plot the image with rectangles around detected faces
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    
    embeddings = []
    face_count = 0
    
    for result in results:
        x, y, width, height = result['box']
        face = rgb_image[y:y + height, x:x + width]
        
        # Ensure the face is not empty
        if face.size == 0:
            continue
        
        # Get face embedding using FaceNet
        face_embedding = embedder.embeddings([face])[0]
        embeddings.append(face_embedding)
        
        # Draw rectangle around the face
        rect = plt.Rectangle((x, y), width, height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

        face_count += 1

    # Display face count
    st.write(f"**Number of faces detected:** {face_count}")
    
    # Show the image with detected faces
    plt.axis('off')
    st.pyplot(plt)

    # Calculate pairwise distances between embeddings for recognition accuracy
    if len(embeddings) > 1:
        st.write("### Similarity Between Faces:")
        threshold = 1.0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                if distance < threshold:
                    st.write(f"✅ Faces **{i + 1}** and **{j + 1}** are likely the same person (distance: {distance:.4f})")
                else:
                    st.write(f"❌ Faces **{i + 1}** and **{j + 1}** are likely different people (distance: {distance:.4f})")

# Streamlit app title
st.title("🧠 Face Detection and Recognition")

# File uploader
uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image button
    if st.button("🚀 Process Image"):
        process_image(image)
