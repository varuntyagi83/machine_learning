import cv2
import numpy as np

# Define a function to read warehouse video stream and identify forklift locations
def identify_forklift_locations(video_stream):
    # Initialize variables
    forklift_locations = []

    # Capture frames from the video stream
    while True:
        # Read a frame from the video stream
        frame = video_stream.read()

        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to binarize the frame
        _, thresh = cv2.threshold(grayscale_frame, 127, 255, cv2.THRESH_BINARY)

        # Apply Canny edge detection to detect edges
        edges = cv2.Canny(thresh, 50, 150)

        # Apply HoughLinesP to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        # Identify forklift locations based on the detected lines
        for line in lines:
            # Extract line endpoints
            x1, y1, x2, y2 = line[0]

            # Check if the line is likely to represent a forklift based on its length and angle
            if (x2 - x1) > 100 and abs((y2 - y1) / (x2 - x1)) > 0.3:
                # Add the midpoint of the line to the list of forklift locations
                midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                forklift_locations.append(midpoint)

        # Draw the identified forklift locations on the frame
        for forklift_location in forklift_locations:
            cv2.circle(frame, forklift_location, 5, (0, 255, 0), -1)

        # Display the frame with identified forklift locations
        cv2.imshow('Forklift Location Tracking', frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the video stream
    video_stream.release()

    # Close all windows
    cv2.destroyAllWindows()

# Example usage
video_path = 'warehouse_video.mp4'
video_stream = cv2.VideoCapture(video_path)

identify_forklift_locations(video_stream)
