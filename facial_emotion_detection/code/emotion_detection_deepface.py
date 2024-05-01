# Make sure to install deepface library using
# pip install deepface

import cv2
from deepface import DeepFace

# Start capturing video from the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = DeepFace.detectFace(frame, detector_backend='opencv')

    # Check if faces are detected
    if faces is not None and len(faces) > 0:
        # Loop through detected faces
        for face in faces:
            (start_x, start_y, end_x, end_y) = (face['box']['x'], face['box']['y'], face['box']['x']+face['box']['w'], face['box']['y']+face['box']['h'])

            # Crop the face from the frame
            face_crop = frame[start_y:end_y, start_x:end_x]

            # Perform emotion prediction
            result = DeepFace.analyze(face_crop, actions=['emotion'])

            # Get the dominant emotion
            emotion = max(result['emotion'], key=result['emotion'].get)

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
