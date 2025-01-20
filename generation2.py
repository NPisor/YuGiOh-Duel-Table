import os
import re
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import logging
from tkinter import Tk, Label, ttk
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

# Configure logging
logging.basicConfig(
    filename="new_generation_with_gui.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Paths and constants
LOG_FILE = "parameter_optimization.log"
CARD_ART_FOLDER = "card_art"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Regular expression to extract parameters and other details
PARAM_REGEX = (
    r"Best params: \{'blur_ksize': (\d+), 'unsharp_weight': ([\d.]+), 'threshold_block_size': (\d+), "
    r"'threshold_c': (\d+), 'canny_thresh1': (\d+), 'canny_thresh2': (\d+)\}, Confidence: ([\d.]+), Detected: (.+)"
)

import queue

class OCRChunkProcessor:
    def __init__(self, master, total_chunks):
        self.master = master
        self.total_chunks = total_chunks
        self.progress_bars = []
        self.labels = []
        self.queue = queue.Queue()  # Queue for thread-safe updates
        self.init_ui()

        # Periodically check the queue for updates
        self.master.after(100, self.process_queue)

    def init_ui(self):
        self.master.title("Chunk Processing Progress")

        for chunk_id in range(self.total_chunks):
            label = Label(self.master, text=f"Chunk {chunk_id + 1}")
            label.grid(row=chunk_id, column=0, sticky="w")
            self.labels.append(label)

            progress = ttk.Progressbar(self.master, orient="horizontal", length=300, mode="determinate")
            progress.grid(row=chunk_id, column=1)
            self.progress_bars.append(progress)

    def process_queue(self):
        """
        Check the queue and update the GUI with any pending updates.
        """
        while not self.queue.empty():
            chunk_id, progress, result = self.queue.get()
            self.update_progress(chunk_id, progress)
            if result:
                self.log_result(chunk_id, result)
        self.master.after(100, self.process_queue)

    def update_progress(self, chunk_id, progress):
        self.progress_bars[chunk_id].config(value=progress)

    def log_result(self, chunk_id, result):
        self.labels[chunk_id].config(text=f"Chunk {chunk_id + 1}: {result}")


# Utility functions
def parse_top_chunks(log_file, top_n=10):
    """
    Parse the top N chunks based on the number of valid detections.
    """
    chunk_results = []
    current_chunk = None
    correct_detections = 0

    with open(log_file, "r") as log:
        for line in log:
            chunk_match = re.search(r"Processing chunk (\d+)", line)
            if chunk_match:
                if current_chunk is not None:
                    chunk_results.append((current_chunk, correct_detections))
                current_chunk = int(chunk_match.group(1))
                correct_detections = 0

            detected_match = re.search(r"Detected: (.+)", line)
            if detected_match:
                detected_text = detected_match.group(1).strip()
                if detected_text and detected_text.lower() != "no valid text detected":
                    correct_detections += 1

    if current_chunk is not None:
        chunk_results.append((current_chunk, correct_detections))

    sorted_chunks = sorted(chunk_results, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:top_n]]

def parse_log_for_params(log_file):
    """
    Parse the log file to extract valid parameters and confidences.
    """
    extracted_params = []

    with open(log_file, "r") as log:
        for line in log:
            match = re.search(PARAM_REGEX, line)
            if match:
                params = tuple(map(float, match.groups()[:6]))
                confidence = float(match.group(7))
                detected_text = match.group(8).strip()

                # Only include valid detections
                if detected_text and confidence > 0:
                    extracted_params.append((params, confidence))

    return extracted_params

def compute_average_params(extracted_params):
    """
    Compute the average parameters based on successful detections.
    """
    if not extracted_params:
        raise ValueError("No valid parameters extracted from the log.")

    params_array = np.array([params for params, _ in extracted_params])
    avg_params = params_array.mean(axis=0)
    return tuple(avg_params)

def preprocess_image(image, blur_ksize, unsharp_weight, threshold_block_size, threshold_c, canny_thresh1, canny_thresh2):
    """
    Preprocess the image using the specified parameters.
    """
    try:
        # Ensure valid threshold_block_size
        threshold_block_size = max(3, int(threshold_block_size))
        if threshold_block_size % 2 == 0:
            threshold_block_size += 1  # Make it odd

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (int(blur_ksize), int(blur_ksize)), 0)
        unsharp = cv2.addWeighted(gray, unsharp_weight, blurred, -1.0 * unsharp_weight, 0)
        thresh = cv2.adaptiveThreshold(
            unsharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            threshold_block_size,
            int(threshold_c),
        )
        edges = cv2.Canny(thresh, int(canny_thresh1), int(canny_thresh2))
        cropped = edges[:100, :]  # Focus on the top region of the card

        # Save preprocessed images for debugging
        debug_output_path = "debug_preprocessed.jpg"
        cv2.imwrite(debug_output_path, cropped)

        return cropped

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise



def perform_ocr(image):
    """
    Perform OCR on the processed image and return detected text and average confidence.
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

def process_chunk(chunk_id, avg_params, processor):
    """
    Process a single chunk of images and update the progress bar.
    """
    files = [
        f for f in os.listdir(CARD_ART_FOLDER) if f.endswith(".jpg")
    ][:200]
    chunk_files = files[chunk_id * 20 : (chunk_id + 1) * 20]

    blur_ksize, unsharp_weight, thresh_block, thresh_c, canny_thresh1, canny_thresh2 = avg_params
    total_files = len(chunk_files)

    for idx, file in enumerate(chunk_files):
        image_path = os.path.join(CARD_ART_FOLDER, file)
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"File not found or unreadable: {file}")
            continue

        processed_image = preprocess_image(image, blur_ksize, unsharp_weight, thresh_block, thresh_c, canny_thresh1, canny_thresh2)
        detected_text, avg_conf = perform_ocr(processed_image)

        expected_name = os.path.splitext(file)[0]
        success = expected_name.lower() in detected_text.lower()
        status = "Success" if success else "Failure"

        # Log results for each file
        logging.info(
            f"Chunk {chunk_id} - File {file} - {status} with confidence {avg_conf:.2f}, Detected: {detected_text}"
        )

        progress = (idx + 1) / total_files * 100
        processor.update_progress(chunk_id, progress)

    processor.log_result(chunk_id, "Completed")

def main():
    top_chunks = parse_top_chunks(LOG_FILE)
    params = parse_log_for_params(LOG_FILE)
    avg_params = compute_average_params(params)
    logging.info(f"Averaged Parameters: {avg_params}")

    root = Tk()
    processor = OCRChunkProcessor(root, len(top_chunks))

    with ThreadPoolExecutor(max_workers=4) as executor:  # Ensure threading runs correctly
        futures = [
            executor.submit(process_chunk, chunk_id, avg_params, processor)
            for chunk_id in range(len(top_chunks))
        ]
        for future in as_completed(futures):  # Wait for threads to complete
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread: {e}")

    root.mainloop()

if __name__ == "__main__":
    main()
