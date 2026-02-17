import cv2
import numpy as np

def image_to_coloring_sheet(
    input_path,
    output_path="coloring_sheet.png",
    target_dpi=300,
    paper_size_inches=(8.5, 11),
    line_thickness=2
):
    """
    Converts any image into a high-quality printable coloring sheet
    without warping aspect ratio.
    """

    # -----------------------------
    # 1. Load Image
    # -----------------------------
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Image not found or incorrect path.")

    original_h, original_w = img.shape[:2]

    # -----------------------------
    # 2. Create Print Canvas
    # -----------------------------
    canvas_w = int(paper_size_inches[0] * target_dpi)
    canvas_h = int(paper_size_inches[1] * target_dpi)

    # Compute scale while preserving aspect ratio
    scale = min(canvas_w / original_w, canvas_h / original_h)

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create white canvas
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Center image
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # -----------------------------
    # 3. Convert to Grayscale
    # -----------------------------
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # 4. Edge-Preserving Smoothing
    # -----------------------------
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # -----------------------------
    # 5. Adaptive Threshold
    # -----------------------------
    thresh = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # -----------------------------
    # 6. Canny Edges
    # -----------------------------
    edges = cv2.Canny(smooth, 100, 200)

    combined = cv2.bitwise_or(thresh, edges)

    # -----------------------------
    # 7. Clean Noise
    # -----------------------------
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # 8. Thicken Lines
    # -----------------------------
    if line_thickness > 0:
        thick_kernel = np.ones((line_thickness, line_thickness), np.uint8)
        cleaned = cv2.dilate(cleaned, thick_kernel, iterations=1)

    # -----------------------------
    # 9. Final Black & White
    # -----------------------------
    final = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)[1]
    final = 255 - final  # white background

    # -----------------------------
    # 10. Save
    # -----------------------------
    cv2.imwrite(output_path, final)

    print(f"Saved print-ready coloring sheet: {output_path}")
    print(f"Final size: {canvas_w} x {canvas_h} (300 DPI)")


if __name__ == "__main__":
    image_to_coloring_sheet(
        input_path="hq720.jpg",
        output_path="printable_coloring_sheet.png",
        line_thickness=2
    )
