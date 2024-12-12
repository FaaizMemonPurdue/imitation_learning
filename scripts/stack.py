import cv2
import numpy as np
import os

def vertically_stack_images(image_paths, output_path):
    images = [cv2.imread(image_path) for image_path in image_paths]
    
    # Check if all images have the same width
    widths = [img.shape[1] for img in images]
    if len(set(widths)) != 1:
        raise ValueError("All images must have the same width to be stacked vertically.")
    
    # Stack images vertically
    stacked_image = np.vstack(images)
    
    # Save the result
    cv2.imwrite(output_path, stacked_image)

if __name__ == "__main__":
    # Example usage
    image_folder = os.environ['HOME'] + "/lstick"
    image_paths = []
    # image_paths.append("expert.png")
    # for i in range(1, 5):
    #     image_paths.append(f"coll{i}.png")
    for i in range(1, 5):
        image_paths.append(f"eval{i}.png")
    image_pacs = [os.path.join(image_folder, i) for i in image_paths]
    output_path = os.path.join(image_folder, "estack.png")
    
    vertically_stack_images(image_pacs, output_path)