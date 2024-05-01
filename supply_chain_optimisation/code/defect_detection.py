# original source: https://github.com/jurgverhoeven/Cook3r
import cv2
import numpy as np

# Define a function to read car image and perform defect detection
def detect_defects(image_path):
    # Read the car image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smoothen the image
    gaussian_blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(gaussian_blur, 50, 150)

    # Apply HoughCircles to detect circular shapes (possible defects)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=30)

    # Draw circles on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Save the image with detected defects
    cv2.imwrite('defect_detected_image.jpg', image)

# Read the car image and perform defect detection
detect_defects('car_image.jpg')
