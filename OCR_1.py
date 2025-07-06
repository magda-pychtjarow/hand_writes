import cv2
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_text_image(image, debug=False):
    """
    Comprehensive preprocessing function for text detection and OCR.

    Args:
        image: PIL Image or numpy array (grayscale or RGB)
        debug: bool, if True shows intermediate steps

    Returns:
        PIL Image: Preprocessed binary image optimized for text detection
    """

    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = np.array(image, dtype=np.uint8)

    # Ensure proper data type
    img_array = img_array.astype(np.uint8)

    # Convert to grayscale if RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    # Ensure uint8 type
    gray = gray.astype(np.uint8)
    original_gray = gray.copy()

    # Step 1: Noise reduction with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 2: Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 3: Deskewing (rotation correction)
    def detect_skew_angle(image):
        # Use Hough Line Transform to detect dominant angle
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                # Focus on near-horizontal lines
                if 85 < angle < 95 or -5 < angle < 5:
                    angles.append(angle if angle < 45 else angle - 90)

            if angles:
                return np.median(angles)
        return 0

    skew_angle = detect_skew_angle(enhanced)
    if abs(skew_angle) > 0.5:  # Only correct if angle is significant
        center = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        deskewed = cv2.warpAffine(enhanced, rotation_matrix,
                                  (enhanced.shape[1], enhanced.shape[0]),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = enhanced

    # Step 4: Advanced binarization using adaptive thresholding
    # Try multiple methods and combine

    # Method 1: Otsu's thresholding
    _, otsu_binary = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 2: Adaptive Gaussian thresholding
    adaptive_gaussian = cv2.adaptiveThreshold(deskewed, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

    # Method 3: Adaptive Mean thresholding
    adaptive_mean = cv2.adaptiveThreshold(deskewed, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 15, 3)

    # Combine methods using weighted average
    combined = cv2.addWeighted(otsu_binary, 0.4, adaptive_gaussian, 0.3, 0)
    combined = cv2.addWeighted(combined, 1.0, adaptive_mean, 0.3, 0)
    _, combined_binary = _, otsu_binary  # cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    # Step 5: Morphological operations to clean up text
    # Create kernels for different operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # Remove small noise
    cleaned1 = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_small)

    # Connect broken characters
    cleaned2 = cv2.morphologyEx(cleaned1, cv2.MORPH_CLOSE, kernel_medium)

    # Fill small gaps in vertical strokes
    cleaned = combined_binary  # cv2.morphologyEx(cleaned1, cv2.MORPH_CLOSE, kernel_line)
    # until here works fine
    # Step 6: Remove small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - cleaned, connectivity=8)

    # Calculate size threshold based on image dimensions
    min_component_size = max(10, (cleaned.shape[0] * cleaned.shape[1]) // 5000)

    # Create mask for components to keep
    mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > min_component_size:
            mask[labels == i] = 255

    # Apply mask (invert because we're working with white text on black background internally)
    final_binary = 255 - mask

    # Step 7: Final cleanup - ensure proper contrast
    final_binary = np.array(cleaned, dtype=np.uint8)
    _, final_binary = cv2.threshold(final_binary, 127, 255, cv2.THRESH_BINARY)

    # Debug visualization
    if debug:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        images = [
            (original_gray, 'Original'),
            (denoised, 'Denoised'),
            (enhanced, 'CLAHE Enhanced'),
            (deskewed, 'Deskewed'),
            (otsu_binary, 'Otsu Binary'),
            (adaptive_gaussian, 'Adaptive Gaussian'),
            (combined_binary, 'Combined Binary'),
            (final_binary, 'Final Result')
        ]

        for i, (img, title) in enumerate(images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Detected skew angle: {skew_angle:.2f} degrees")
        print(f"Minimum component size threshold: {min_component_size} pixels")
        print(f"Removed {num_labels - 1 - np.sum(np.unique(mask) == 255)} small components")

    # Convert back to PIL Image
    return final_binary


def preprocess_for_ocr(image, line_height=127, target_height=None, debug=False):
    """
    Specialized preprocessing for OCR with optional resizing.

    Args:
        image: PIL Image or numpy array
        debug: bool, show debug information

    Returns:
        PIL Image: Preprocessed image optimized for OCR
    """
    print("Additional pp")
    # First apply general preprocessing
    processed = preprocess_text_image(image, debug)
    processed_array = np.array(processed)

    if debug:
        print(f"Final image size: {processed_array.shape}")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image), cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(processed_array, cmap='gray')
        plt.title('Preprocessed for OCR')
        plt.axis('off')
        plt.show()

    return processed_array


def read_with_easyocr(image_path):
    import cv2
    import easyocr
    # Initialize EasyOCR with English
    reader = easyocr.Reader(['en'])

    # Load the image
    # [LOG] adding preprocessing didnt improve the detection
    image = preprocess_for_ocr(cv2.imread(image_path), False)

    # Using easyocr for segmentation
    results = reader.readtext(image, detail=1)

    lines = []
    for (bbox, text, conf) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(bottom_right[0], top_right[0]))
        y_max = int(max(bottom_right[1], bottom_left[1]))

        line_img = image[y_min:y_max, x_min:x_max]
        lines.append((line_img, text, conf))
    print("Detected ", len(lines), " lines.")
    # Display segmented lines
    # for i, (img, text, conf) in enumerate(lines):
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.title(f'Line {i+1}: {text}')
    #     plt.axis('off')
    #     plt.show()

    words = [(img, word, conf) for [img, word, conf] in lines]

    return words
