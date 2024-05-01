import cv2
import cvlib as cv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np

# Function to generate synthetic dataset
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    emotions = ['happy', 'sad', 'angry', 'neutral']
    data = {'image': [], 'emotion': []}

    for _ in range(num_samples):
        # Generate random synthetic image data (replace this with your actual image data loading)
        image_data = np.random.rand(100, 100, 3) * 255  # Example random image
        emotion = np.random.choice(emotions)
        data['image'].append(image_data)
        data['emotion'].append(emotion)

    df = pd.DataFrame(data)
    return df

# Generate synthetic training data
train_data = generate_synthetic_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data['image'], train_data['emotion'], test_size=0.2, random_state=42)

# Build a simple pipeline with a classifier
model = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100))  # Use RandomForestClassifier for image data
])

# Flatten the images for training
X_train_flat = [img.flatten() for img in X_train]

# Train the model
model.fit(X_train_flat, y_train)

# Save the trained model
model_filename = 'emotion_detection_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved to {model_filename}")

# Load the trained model
loaded_model = joblib.load(model_filename)

# Start capturing video from the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)

    # Loop through detected faces
    for face, confidence in zip(faces, confidences):
        (start_x, start_y, end_x, end_y) = face

        # Crop the face from the frame
        face_crop = frame[start_y:end_y, start_x:end_x]

        # Resize the face for prediction (adjust the size as needed)
        face_resize = cv2.resize(face_crop, (100, 100))

        # Flatten the face image for prediction
        face_flat = face_resize.flatten()

        # Perform emotion prediction
        emotion = loaded_model.predict([face_flat])[0]

        # Draw bounding box and label on the frame
        label = f'Emotion: {emotion}'
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
