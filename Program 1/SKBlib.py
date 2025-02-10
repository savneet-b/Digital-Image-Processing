import cv2
import numpy as np

def histogram_equalization(img):
    # Convert the image to grayscale if it is in color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = img.shape

    # Compute the histogram of pixel values
    hist = [0] * 256
    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            hist[pixel_value] += 1

    # Compute the cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Normalize the CDF
    cdf_min = min(val for val in cdf if val > 0)  # Find the minimum non-zero CDF value
    cdf_max = cdf[-1]
    cdf_normalized = [(val - cdf_min) * 255 / (cdf_max - cdf_min) for val in cdf]

    # Apply histogram equalization
    img_equalized = np.zeros_like(img)  # Create an array with the same shape as the input image
    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            img_equalized[i, j] = int(cdf_normalized[pixel_value])

    return img_equalized

def floodfill_frontier(img, seed_point, new_value):
    # Function to perform flood fill algorithm using a frontier approach
    height, width = img.shape
    original_value = img[seed_point]
    frontier = [seed_point]

    while frontier:
        x, y = frontier.pop()
        if img[x, y] == original_value:
            img[x, y] = new_value
            # Add neighboring pixels to the frontier
            if x > 0: frontier.append((x - 1, y))  # Up
            if x < height - 1: frontier.append((x + 1, y))  # Down
            if y > 0: frontier.append((x, y - 1))  # Left
            if y < width - 1: frontier.append((x, y + 1))  # Right

def floodfill_separate(img, seed_point, new_value):
    # Function to perform flood fill and return a copy of the modified image
    img_copy = img.copy()
    floodfill_frontier(img_copy, seed_point, new_value)
    return img_copy

def simple_threshold(img, threshold):
    # Function to apply a simple thresholding operation
    output_img = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] > threshold:
                output_img[x, y] = 255  # Set pixel to white if above the threshold
            else:
                output_img[x, y] = 0  # Set pixel to black otherwise
    return output_img

def double_threshold(img, low_thresh, high_thresh):
    # Function to apply double thresholding
    height, width = img.shape
    output = np.zeros_like(img, dtype=np.uint8)

    # Create masks for high and low thresholds
    high_mark = img > high_thresh
    low_mark = (img > low_thresh) & ~high_mark

    # Set output image based on thresholds
    output[high_mark] = 180  # High threshold regions are white
    output[low_mark] = 100   # Low threshold regions are gray
    output[img <= low_thresh] = 0  # Below low threshold regions are black

    return output

def erosion(img, kernel):
    # Function to perform erosion operation
    height, width = img.shape
    k_height, k_width = kernel.shape
    output = np.ones_like(img, dtype=np.uint8) * 255  # Start with all white

    # Loop through image pixels (excluding border pixels)
    for x in range(k_height // 2, height - k_height // 2):
        for y in range(k_width // 2, width - k_width // 2):
            all_on = True
            for x_k in range(k_height):
                for y_k in range(k_width):
                    if kernel[x_k, y_k] and img[x + x_k - k_height // 2, y + y_k - k_width // 2] != 255:
                        all_on = False
                        break
            output[x, y] = 255 if all_on else 0  # Set to white if all kernel elements match, else black
    return output

def dilation(img, kernel):
    # Function to perform dilation operation
    height, width = img.shape
    k_height, k_width = kernel.shape
    output = np.zeros_like(img, dtype=np.uint8)  # Start with all black

    # Loop through image pixels (excluding border pixels)
    for x in range(k_height // 2, height - k_height // 2):
        for y in range(k_width // 2, width - k_width // 2):
            any_on = False
            for x_k in range(k_height):
                for y_k in range(k_width):
                    if kernel[x_k, y_k] and img[x + x_k - k_height // 2, y + y_k - k_width // 2] == 255:
                        any_on = True
                        break
            output[x, y] = 255 if any_on else 0  # Set to white if any kernel elements match, else black
    return output
