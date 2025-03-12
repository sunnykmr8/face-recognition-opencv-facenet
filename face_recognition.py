import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# Initialize MTCNN detector
detector = MTCNN()

# Initialize FaceNet model
embedder = FaceNet()

# Read the image
image_path = '/content/1200px-Which_friend_are_you.png'  # Replace with your image path
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = detector.detect_faces(rgb_image)

# Plot the image with detected faces
plt.imshow(rgb_image)
ax = plt.gca()

# Initialize face count
face_count = 0

# Extract embeddings and draw rectangles
for result in results:
    x, y, width, height = result['box']
    face = rgb_image[y:y+height, x:x+width]
    face_embedding = embedder.embeddings([face])
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
