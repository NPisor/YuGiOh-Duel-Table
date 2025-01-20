import cv2
import pytesseract
import os
from pytesseract import Output
from statistics import mean
import logging
import random
import numpy as np

# Configure logging
logging.basicConfig(
    filename="continuous_optimization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Card art folder
CARD_ART_FOLDER = "results"

# Initial parameters
PARAMS = {
    "blur_ksize": 11,
    "unsharp_weight": 2.69,
    "threshold_block_size": 1,
    "threshold_c": 1,
    "canny_thresh1": 180,
    "canny_thresh2": 260,
}

# Increment steps for each parameter
INCREMENTS = {
    "blur_ksize": 2,  # Must remain odd
    "unsharp_weight": 0.1,
    "threshold_block_size": 2,  # Must remain odd
    "threshold_c": 0,
    "canny_thresh1": 5,
    "canny_thresh2": 5,
}

def preprocess_image(image, params):
    """
    Preprocess the image for OCR using sharpening, thresholding, edge detection,
    and enhanced noise suppression.
    """
    # Convert to grayscale
    if len(image.shape) == 3:  # Check if the image is in color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale, skip conversion

    # Resize the image (scale to 1191x2000)
    resized = cv2.resize(gray, (1191, 2000), interpolation=cv2.INTER_LINEAR)

    # Define ROI (upper part of the card for text)
    roi = resized[20:250, 20:1000]

    # Apply CLAHE for dynamic contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(resized)

    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)

    # Ensure valid threshold_block_size (must be odd and >1)
    threshold_block_size = params["threshold_block_size"]
    if threshold_block_size % 2 == 0:  # Make it odd if even
        threshold_block_size += 1
    if threshold_block_size <= 1:  # Ensure it is greater than 1
        threshold_block_size = 3

    # Adaptive thresholding for binary image
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        threshold_block_size, params["threshold_c"]
    )

    # Apply Canny edge detection
    edges = cv2.Canny(thresholded, params["canny_thresh1"], params["canny_thresh2"])

    # Morphological operations to refine edges
    kernel_open = np.ones((3, 3), np.uint8)  # Kernel for opening (noise removal)
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = np.ones((3, 3), np.uint8)  # Kernel for dilation (thickening text)
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)

    # Combine thresholded and dilated edges for better OCR input
    combined = cv2.addWeighted(thresholded, 0.5, dilated, 0.5, 0)

    # Save processed image for debugging
    cv2.imwrite("processed_final_refined.png", combined)

    return combined



def perform_ocr(image):
    """
    Perform OCR and return the detected text and average confidence.
    """
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    detected_text = " ".join(
        [
            ocr_data["text"][i]
            for i in range(len(ocr_data["text"]))
            if int(ocr_data["conf"][i]) > 0
        ]
    )
    confidence_scores = [
        int(ocr_data["conf"][i]) for i in range(len(ocr_data["conf"])) if int(ocr_data["conf"][i]) > 0
    ]
    avg_confidence = mean(confidence_scores) if confidence_scores else 0
    return detected_text.strip(), avg_confidence

def refine_parameters(image, expected_name):
    """
    Refine parameters for a single image.
    """
    global PARAMS
    best_params = PARAMS.copy()
    best_confidence = 0
    detected_name = ""

    for param in PARAMS.keys():
        current_value = PARAMS[param]
        step = INCREMENTS[param]

        for direction in [-1, 1]:  # Test decrement and increment
            PARAMS[param] = current_value + (step * direction)

            # Ensure valid ranges for parameters
            if param == "blur_ksize" and PARAMS[param] % 2 == 0:
                continue  # Blur kernel size must be odd
            if PARAMS[param] <= 0:
                continue  # Parameters must remain positive

            # Preprocess and perform OCR
            processed_image = preprocess_image(image, PARAMS)
            detected_text, avg_conf = perform_ocr(processed_image)

            # Display the current processed image for feedback
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(1)

            # Evaluate
            if expected_name.lower() in detected_text.lower() and avg_conf > best_confidence:
                best_confidence = avg_conf
                best_params[param] = PARAMS[param]
                detected_name = detected_text

            # Revert to the original value if performance worsens
            PARAMS[param] = current_value

    logging.info(f"Best params: {best_params}, Confidence: {best_confidence}, Detected: {detected_name}")
    PARAMS = best_params

def process_webcam():
    """
    Process frames from the webcam.
    """
    try:
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Define expected name for debugging (this could be replaced with a dynamic approach)
            expected_name = "WebcamFrame"

            # Preprocess and refine parameters for the current frame
            processed_image = preprocess_image(frame, PARAMS)
            detected_text, avg_conf = perform_ocr(processed_image)

            # Display the processed image and OCR result
            cv2.imshow("Processed Image", processed_image)
            print(f"Detected Text: {detected_text}, Confidence: {avg_conf:.2f}")

            # Press 'q' to quit the webcam feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Webcam processing interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    logging.info("Starting webcam OCR processing...")
    process_webcam()
    logging.info("Webcam OCR processing completed.")


def process_images():
    """
    Process images continuously.
    """
    try:
        while True:  # Infinite loop until manually stopped
            logging.info("Starting a new optimization iteration...")
            files = [f for f in os.listdir(CARD_ART_FOLDER) if f.endswith(".png")]
            selected_files = random.sample(files, min(len(files), 100))  # Select up to 100 random files

            for filename in selected_files:
                image_path = os.path.join(CARD_ART_FOLDER, filename)
                image = cv2.imread(image_path)
                expected_name = os.path.splitext(filename)[0]

                # Refine parameters for the current image
                refine_parameters(image, expected_name)

            logging.info(f"Completed an iteration with parameters: {PARAMS}")

    except KeyboardInterrupt:
        logging.info("Optimization process stopped by user.")
        cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    logging.info("Starting continuous optimization...")
    process_webcam()
    logging.info("Optimization completed.")
