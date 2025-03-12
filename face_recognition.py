import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import matplotlib.pyplot as plt
import numpy as np

# Initialize MTCNN detector
detector = MTCNN()

# Initialize FaceNet model
embedder = FaceNet()

# Read the image
image_path = 'IMG/1200px-Which_friend_are_you.png'  # Replace with your image path

image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = detector.detect_faces(rgb_image)

# Plot the image with detected faces
plt.imshow(rgb_image)
ax = plt.gca()

# Initialize face count
face_count = 0

# List to store embeddings
embeddings = []

# Extract embeddings and draw rectangles
for result in results:
    x, y, width, height = result['box']
    face = rgb_image[y:y+height, x:x+width]
    face_embedding = embedder.embeddings([face])
    embeddings.append(face_embedding)
    print("Face Embedding:", face_embedding)

    # Draw rectangle around the face
    rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)

    # Increment face count
    face_count += 1

# Display face count
print(f"Number of faces detected: {face_count}")

plt.axis('off')
plt.show()

# Calculate pairwise distances between embeddings for recognition accuracy
if len(embeddings) > 1:
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"Distance between face {i+1} and face {j+1}: {distance}")

# Example threshold for recognition (you may need to adjust this based on your use case)
threshold = 1.0
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        distance = np.linalg.norm(embeddings[i] - embeddings[j])
        if distance < threshold:
            print(f"Faces {i+1} and {j+1} are likely the same person (distance: {distance})")
        else:
            print(f"Faces {i+1} and {j+1} are likely different people (distance: {distance})")
