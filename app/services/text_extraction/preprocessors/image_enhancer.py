# preprocessors/image_enhancer.py
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Preprocess images to improve OCR accuracy.

    Applies various image processing techniques:
    - Grayscale conversion
    - Noise reduction
    - Contrast enhancement
    - Binarization (thresholding)
    - Deskewing (rotation correction)
    - Border removal
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize image enhancer.

        Args:
            config: Optional configuration dictionary with:
                - denoise_strength: Denoising intensity (default: 10)
                - contrast_factor: Contrast enhancement factor (default: 1.5)
                - deskew: Enable deskewing (default: True)
        """
        self.config = config or {}
        self.denoise_strength = self.config.get('denoise_strength', 10)
        self.contrast_factor = self.config.get('contrast_factor', 1.5)
        self.deskew = self.config.get('deskew', True)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline to image.

        Args:
            image: Input image as numpy array (BGR or grayscale)

        Returns:
            Enhanced image ready for OCR
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply enhancement steps
        enhanced = self._remove_noise(gray)
        enhanced = self._enhance_contrast(enhanced)
        enhanced = self._binarize(enhanced)

        if self.deskew:
            enhanced = self._deskew_image(enhanced)

        enhanced = self._remove_borders(enhanced)

        return enhanced

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise using non-local means denoising.

        This algorithm is effective for removing Gaussian noise
        while preserving edges and text details.
        """
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE works on small regions (tiles) rather than the whole image,
        making it effective for documents with varying lighting.
        """
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(image)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to binary (black and white) using adaptive thresholding.

        Adaptive thresholding calculates threshold for small regions,
        handling documents with shadows or uneven lighting.
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Size of pixel neighborhood
            C=2  # Constant subtracted from mean
        )

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image rotation/skew.

        Uses Hough Line Transform to detect dominant line angles
        and rotates image to align text horizontally.
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return image

        # Calculate average angle of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider nearly horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return image

        # Use median angle to avoid outliers
        median_angle = np.median(angles)

        # Only rotate if skew is significant
        if abs(median_angle) < 0.5:
            return image

        # Rotate image to correct skew
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
        return rotated

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders from scanned documents.

        Finds the bounding box of content and crops to that region.
        """
        # Find contours
        contours, _ = cv2.findContours(
            255 - image,  # Invert so text is white
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find bounding box of all contours
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Add small padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        # Only crop if we found meaningful bounds
        if x_max > x_min and y_max > y_min:
            return image[y_min:y_max, x_min:x_max]

        return image

    def estimate_quality(self, image: np.ndarray) -> float:
        """
        Estimate image quality for OCR suitability.

        Considers:
        - Resolution (DPI estimation)
        - Contrast ratio
        - Noise level

        Args:
            image: Input image

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resolution score (based on image size)
        height, width = gray.shape
        resolution_score = min((height * width) / (1000 * 1000), 1.0)

        # Contrast score
        contrast = gray.std()
        contrast_score = min(contrast / 60, 1.0)

        # Noise estimation using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # High variance indicates sharp edges (good), very high indicates noise
        noise_score = min(laplacian_var / 500, 1.0) if laplacian_var < 2000 else 0.5

        return (resolution_score * 0.3 + contrast_score * 0.4 + noise_score * 0.3)