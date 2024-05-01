import cv2
import numpy as np

# Define a function to read shipping container image and identify product locations
def identify_product_locations(image_path):
    # Read the shipping container image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours of product boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the location of each product box
    product_locations = []
    for contour in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Add the bounding box coordinates to the list of product locations
        product_locations.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })

    return product_locations

# Read the shipping container image
image_path = 'shipping_container_image.jpg'
product_locations = identify_product_locations(image_path)

# Print the locations of all product boxes
for product_location in product_locations:
    print(f"Product box location: ({product_location['x']}, {product_location['y']}), dimensions: ({product_location['w']},{product_location['h']})")
