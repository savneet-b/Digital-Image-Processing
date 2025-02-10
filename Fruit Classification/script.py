import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from SKBlib import clean_threshold_image, connected_components, region_properties, PCA, wall_following, \
    classify_objects, get_threshold_values, draw_axes

def main():
    root = tk.Tk()  # Create a Tkinter root window
    root.withdraw()

    # Ask the user to select the image file
    img_path = filedialog.askopenfilename(
        title="Select the Image",  # Title of the file dialog
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png")]  # Allowed image file types
    )

    if not img_path:  # Check if the user selected a file
        print("No file selected. Exiting...")  # Print message if no file is selected
        return  # Exit the function

    # Load the original image for later visualization
    original_img = cv2.imread(img_path)  # Read the selected image using OpenCV

    # Retrieve threshold values from SKBlib
    low_thresh, high_thresh = get_threshold_values()  # Get the threshold values for image processing

    # Step 1: Clean threshold image
    clean_img = clean_threshold_image(img_path, low_thresh, high_thresh)  # Apply thresholding to the image

    # Display the clean thresholded image
    cv2.imshow('Clean Thresholded Image', clean_img)  # Show the thresholded image in a window

    # Step 2: Connected components
    labeled_img = connected_components(clean_img)  # Identify connected components in the thresholded image

    # Display the connected components image
    cv2.imshow('Connected Components', labeled_img)  # Show the labeled image in a window

    # Create a color image for classification (to overlay later)
    classified_img = np.zeros_like(original_img)  # Create a blank image with the same shape as the original

    # Step 3: Find contours (regions) for each object
    contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image

    for contour in contours:  # Iterate over each contour found
        # Create a mask for the current region
        mask = np.zeros_like(clean_img)  # Create a blank mask for the current contour
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the contour on the mask

        # Step 3: Compute region properties
        moments, area = region_properties(mask)  # Calculate the moments and area of the current contour

        if area == 0:  # Check if the area is zero
            continue  # Skip to the next contour if the area is zero

        # Step 4: Perform PCA on the moments
        properties = PCA(moments)  # Perform Principal Component Analysis (PCA) on the calculated moments
        eigenvalues, theta, major_axis_length, minor_axis_length, eccentricity = properties  # Unpack PCA results

        # Get the center of the region
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))  # Calculate the center of the region

        # Step 6: Classify the object based on its properties
        fruit_type, color = classify_objects(properties)  # Classify the fruit type and its color based on properties
        print(f"Classified object as {fruit_type}")  # Print the classified object type

        # Draw the major and minor axes in the respective color, with a specified scale
        draw_axes(classified_img, center, theta, major_axis_length, minor_axis_length, color=color, scale_factor=0.5)  # Draw axes on the classified image

        # Draw the perimeter in the specified color (red, orange, yellow)
        cv2.drawContours(classified_img, [contour], -1, color, thickness=2)  # Draw the contour on the classified image

        # Draw the center of the object as a small black circle
        cv2.circle(classified_img, center, 5, (0, 0, 0), -1)  # Draw a small black circle at the center of the object

    # Step 7: Overlay classified objects image onto the original image
    overlay_img = cv2.addWeighted(original_img, 0.7, classified_img, 0.3, 0)  # Blend the original image and classified image

    # Display the original image
    cv2.imshow('Original Image', original_img)  # Show the original image in a window

    # Display the final overlaid image with classifications
    cv2.imshow('Classified Objects Overlay', overlay_img)  # Show the overlaid image in a window

    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Execute the main function if this script is run directly
