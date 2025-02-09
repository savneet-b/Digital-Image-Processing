import cv2
import numpy as np
import matplotlib.pyplot as plt
import SKBlib as lib  # Import helper functions from SKBlib

def test_histogram_equalization(img_path):
    """
    Test the histogram equalization function by applying it to a grayscale image and displaying the result.
    :param img_path: Path to the image file to be processed.
    """
    # Load the image in grayscale mode directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Apply histogram equalization using the function from SKBlib
    equalized_img = lib.histogram_equalization(img)

    # Create a figure to display the original and equalized images side by side
    plt.figure(figsize=(12, 6))

    # Display the original grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')  # Title for the original image subplot

    # Display the histogram-equalized image
    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image')  # Title for the equalized image subplot

    plt.show()  # Render the plots

def test_floodfill_frontier(img_path, seed_point):
    """
    Test the floodfill algorithm using a frontier approach. Modifies the input image directly.
    :param img_path: Path to the image file where floodfill will be applied.
    :param seed_point: Tuple (x, y) representing the starting point for floodfill.
    """
    # Load the image in grayscale mode directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Apply floodfill using the frontier method from SKBlib. Set floodfilled area to white (255)
    lib.floodfill_frontier(img, seed_point, new_value=255)

    # Display the image after floodfill operation
    plt.imshow(img, cmap='gray')
    plt.title('Floodfill (Frontier)')  # Title for the floodfill image
    plt.show()  # Render the plot

def test_floodfill_separate(img_path, seed_point):
    """
    Test the floodfill algorithm with a separate output image, leaving the original image unchanged.
    :param img_path: Path to the image file where floodfill will be applied.
    :param seed_point: Tuple (x, y) representing the starting point for floodfill.
    """
    # Load the image in grayscale mode directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Apply floodfill and obtain a new image where floodfill is performed. Floodfilled area is set to gray (128)
    output_img = lib.floodfill_separate(img, seed_point, new_value=128)

    # Display the new image with floodfill applied
    plt.imshow(output_img, cmap='gray')
    plt.title('Floodfill (Separate Output)')  # Title for the separate output image
    plt.show()  # Render the plot

def test_double_threshold(img_path):
    """
    Test the double thresholding function by applying it to a grayscale image and displaying the result.
    :param img_path: Path to the image file to be processed.
    """
    # Load the image in grayscale mode directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Apply double thresholding with specified low and high thresholds
    thresholded_img = lib.double_threshold(img, low_thresh=100, high_thresh=180)

    # Display the image after double thresholding
    plt.imshow(thresholded_img, cmap='gray')
    plt.title('Double Thresholding')  # Title for the thresholded image
    plt.show()  # Render the plot

def test_erosion_dilation(img_path):
    """
    Test the erosion and dilation functions on a grayscale image and display the results.
    :param img_path: Path to the image file to be processed.
    """
    # Load the image in grayscale mode directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Define a 3x3 square kernel for morphological operations
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Apply erosion and dilation using the kernel and functions from SKBlib
    eroded_img = lib.erosion(img, kernel)
    dilated_img = lib.dilation(img, kernel)

    # Create a figure to display the results of erosion and dilation side by side
    plt.figure(figsize=(12, 6))

    # Display the eroded image
    plt.subplot(1, 2, 1)
    plt.imshow(eroded_img, cmap='gray')
    plt.title('Eroded Image')  # Title for the eroded image subplot

    # Display the dilated image
    plt.subplot(1, 2, 2)
    plt.imshow(dilated_img, cmap='gray')
    plt.title('Dilated Image')  # Title for the dilated image subplot

    plt.show()  # Render the plots

def main():
    """
    Main function to interactively select and run different image processing tests.
    Prompts the user to input the image path and select which test to run.
    """
    print("Select a test to run:")
    print("1. Histogram Equalization")
    print("2. Floodfill using a frontier (Modifies image)")
    print("3. Floodfill with separate output (Does not modify image)")
    print("4. Double Thresholding")
    print("5. Erosion and Dilation")

    # Get the user's choice of test
    choice = input("Enter the test number (1-5): ")

    # Prompt for the path to the image file
    img_path = input("Enter the path to the image: ")

    # Run the selected test based on user choice
    if choice == '1':
        test_histogram_equalization(img_path)
    elif choice == '2':
        # Get the seed point coordinates for floodfill
        x = int(input("Enter the seed point x-coordinate: "))
        y = int(input("Enter the seed point y-coordinate: "))
        test_floodfill_frontier(img_path, (x, y))
    elif choice == '3':
        # Get the seed point coordinates for floodfill
        x = int(input("Enter the seed point x-coordinate: "))
        y = int(input("Enter the seed point y-coordinate: "))
        test_floodfill_separate(img_path, (x, y))
    elif choice == '4':
        test_double_threshold(img_path)
    elif choice == '5':
        test_erosion_dilation(img_path)
    else:
        print("Invalid choice. Please select a valid test number (1-5).")

if __name__ == "__main__":
    main()
