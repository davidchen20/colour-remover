import cv2
import numpy as np
import os
# from hed_colourify import create_hed_coloring_sheet

def colouring_imageify(input_image_path):
    img = cv2.imread(input_image_path)

    if img is None:
        raise ValueError("Image not found.")
    
    # Remove any colour from the image
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(100,100))
    enhanced_grey = clahe.apply(grey)

    # smooth = cv2.bilateralFilter(grey, 9, 75, 75)
    # smooth = cv2.GaussianBlur(grey, (5, 5), 0)
    smooth = cv2.bilateralFilter(enhanced_grey, 9, 75, 75)

    # Extract edges from the greyscale image
    edges = cv2.Canny(smooth, 30, 100)

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    clean_canvas = np.ones_like(img) * 255

    # Find the "paths" of the edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.arcLength(cnt, True) < 15:
            continue

        # Calculate how much to "straighten" the line
        # Increase the 0.01 factor for straighter/simpler lines
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Draw the straightened lines onto our white canvas
        cv2.drawContours(clean_canvas, [approx], -1, (0), thickness=1)

    # colouring = 255 - clean_canvas

    filename = os.path.basename(input_image_path)
    name_part = os.path.splitext(filename)[0]
    output_filename = name_part + "_out.png"

    output_dir = "output_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    output_image_path = os.path.join(output_dir, output_filename)
    # cv2.imwrite(output_image_path, clean_canvas)
    cv2.imwrite(output_image_path, edges)
    # cv2.imwrite(output_image_path, enhanced_grey)

if __name__ == "__main__":
    colouring_imageify("input_images/test1.jpg")