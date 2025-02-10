import cv2
import numpy as np

def clean_threshold_image(image_path, low_thresh, high_thresh):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a binary threshold to the image:
    # Pixels with intensity greater than low_thresh are set to white (255),
    # and those below high_thresh are set to black (0).
    _, binary_img = cv2.threshold(img, low_thresh, high_thresh, cv2.THRESH_BINARY)

    # Return the binary image where fruits are white (foreground) and the background is black
    return binary_img

def get_threshold_values():
    # Define and return the threshold values for the binary thresholding process
    low_thresh = 150  # Lower threshold value
    high_thresh = 255  # Upper threshold value
    return low_thresh, high_thresh

# Step 2: Connected Components
def connected_components(binary_image):
    # Use OpenCV's connectedComponents function to label connected regions in the binary image
    num_labels, labels = cv2.connectedComponents(binary_image)
    # Create a blank label image with the same shape as the binary image
    label_image = np.zeros_like(binary_image, dtype=np.uint8)

    # Iterate through each label (excluding background label which is 0)
    for label in range(1, num_labels):
        # Scale the label value to fit into the range of 0-255 for visualization
        label_intensity = (label * 40) % 256  # Ensure intensity is capped at 255
        # Assign the scaled intensity to the label image where the label matches
        label_image[labels == label] = label_intensity

    # Return the labeled image for visualization of connected components
    return label_image

# Step 3: Region Properties
def region_properties(binary_image):
    # Calculate image moments for the binary image, which are useful for shape analysis
    moments = cv2.moments(binary_image)
    # The area is represented by the zeroth moment
    area = moments['m00']  # Area of the detected region
    # Return both the moments and the area for further analysis
    return moments, area

# Step 4: PCA (Principal Component Analysis)
def PCA(moments):
    # Extract second-order central moments from the moments dictionary
    mu20 = moments['mu20']
    mu02 = moments['mu02']
    mu11 = moments['mu11']

    # Create the covariance matrix using the central moments
    cov_matrix = np.array([[mu20, mu11], [mu11, mu02]])
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Ensure all eigenvalues are non-negative for physical relevance
    eigenvalues = np.clip(eigenvalues, 0, None)

    # Calculate the lengths of the major and minor axes based on eigenvalues
    major_axis_length = np.sqrt(np.max(eigenvalues))
    minor_axis_length = np.sqrt(np.min(eigenvalues))

    # Calculate eccentricity of the shape based on axis lengths
    if major_axis_length > 0:
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
    else:
        eccentricity = 0  # Handle case where major axis length is zero

    # Calculate the orientation of the shape (angle) using arctangent
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Return the computed eigenvalues, orientation angle, axis lengths, and eccentricity
    return eigenvalues, theta, major_axis_length, minor_axis_length, eccentricity

# Step 5: Wall-following Algorithm
def wall_following(binary_image):
    # Create an empty image to draw the perimeter contours
    perimeter_image = np.zeros_like(binary_image)
    # Find contours of the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the perimeter image in white
    cv2.drawContours(perimeter_image, contours, -1, 255, 1)  # -1 means draw all contours
    # Return the image with drawn perimeters
    return perimeter_image

# Step 6: Classification of Objects
def classify_objects(properties):
    # Unpack properties such as major axis length and eccentricity
    _, _, major_axis_length, _, eccentricity = properties
    # Classify the object based on its shape and size
    if eccentricity > 0.8:  # High eccentricity indicates elongated shape (banana)
        return 'banana', (0, 255, 255)  # Return label and yellow color for banana
    elif major_axis_length < 50:  # Small size indicates round shape (tangerine)
        return 'tangerine', (0, 165, 255)  # Return label and orange color for tangerine
    else:
        return 'apple', (0, 0, 255)  # Return label and red color for apple

# Step 7: Draw axes (Major and Minor axes)
def draw_axes(image, center, theta, major_axis_length, minor_axis_length, color=(0, 0, 0), scale_factor=0.5):
    # Define the center point of the shape
    x_center, y_center = int(center[0]), int(center[1])

    # Calculate the actual lengths of the major and minor axes using a scale factor
    length_major = major_axis_length * scale_factor
    length_minor = minor_axis_length * scale_factor

    # Calculate endpoints of the major axis using trigonometric functions
    x_major = int(length_major * np.cos(theta))
    y_major = int(length_major * np.sin(theta))

    # Calculate endpoints of the minor axis (perpendicular to major axis)
    x_minor = int(length_minor * np.cos(theta + np.pi / 2))
    y_minor = int(length_minor * np.sin(theta + np.pi / 2))

    # Draw the major axis on the image in the specified color
    cv2.line(image, (x_center - x_major, y_center - y_major), (x_center + x_major, y_center + y_major), color, 2)

    # Draw the minor axis on the image in the specified color
    cv2.line(image, (x_center - x_minor, y_center - y_minor), (x_center + x_minor, y_center + y_minor), color, 2)

    # Return the image with drawn axes
    return image
