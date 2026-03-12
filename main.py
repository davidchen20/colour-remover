import cv2
import os

def colouring_imageify(input_image_path):
    img = cv2.imread(input_image_path)

    if img is None:
        raise ValueError("Image not found.")
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filename = os.path.basename(input_image_path)
    name_part = os.path.splitext(filename)[0]
    output_filename = name_part + "_out.png"

    output_dir = "output_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    output_image_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_image_path, grey)


if __name__ == "__main__":
    colouring_imageify("input_images/test1.jpg")