import cv2
import numpy as np


def image_to_clean_coloring_sheet(
    input_path,
    output_path="coloring_sheet.png",
    target_dpi=300,
    paper_size_inches=(8.5, 11),
    simplify_strength=10,
    line_thickness=2
):
    """
    Generates a simplified coloring sheet with more white space.
    """

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Image not found.")

    h, w = img.shape[:2]

    # -----------------------------
    # Resize to print canvas (no warp)
    # -----------------------------
    canvas_w = int(paper_size_inches[0] * target_dpi)
    canvas_h = int(paper_size_inches[1] * target_dpi)

    scale = min(canvas_w / w, canvas_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # -----------------------------
    # 1. Heavy Smoothing (removes texture)
    # -----------------------------
    smooth = cv2.bilateralFilter(canvas, 15, 150, 150)
    smooth = cv2.medianBlur(smooth, 7)

    # -----------------------------
    # 2. Color Quantization (reduces regions)
    # -----------------------------
    Z = smooth.reshape((-1, 3))
    Z = np.float32(Z)

    K = simplify_strength  # fewer clusters = bigger regions
    _, label, center = cv2.kmeans(
        Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    quantized = center[label.flatten()]
    quantized = quantized.reshape((smooth.shape))

    # -----------------------------
    # 3. Edge Detection on Simplified Image
    # -----------------------------
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)

    # -----------------------------
    # 4. Connect Lines
    # -----------------------------
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -----------------------------
    # 5. Thicken Lines
    # -----------------------------
    if line_thickness > 0:
        thick_kernel = np.ones((line_thickness, line_thickness), np.uint8)
        edges = cv2.dilate(edges, thick_kernel, iterations=1)

    # -----------------------------
    # 6. Create Final Sheet
    # -----------------------------
    final = 255 - edges  # white background

    cv2.imwrite(output_path, final)
    print(f"Saved simplified coloring sheet: {output_path}")



if __name__ == "__main__":
    image_to_clean_coloring_sheet(
        input_path="ithaca.jpg",
        output_path="ithacaColoringSheet.png",
        simplify_strength=8,
        line_thickness=2
    )
