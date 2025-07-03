import os
import shutil
import cv2
import cvlib as cv
from PIL import Image
import numpy as np

def analyze_image(image_path):
    """
    Analyzes an image and returns a dictionary of its properties.
    """
    try:
        with Image.open(image_path) as img:
            grayscale_img = img.convert('L')
            brightness = np.mean(np.array(grayscale_img))
            contrast = np.std(np.array(grayscale_img))
            laplacian = np.abs(np.array(grayscale_img, dtype=np.float64) - np.roll(np.roll(np.array(grayscale_img, dtype=np.float64), 1, axis=0), 1, axis=1))
            sharpness = np.var(laplacian)
            return {'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness}
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def is_blank(image_path):
    """
    Checks if an image is blank (contains no faces).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return True # Treat unreadable images as blank
        faces, confidences = cv.detect_face(img)
        return len(faces) == 0
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return True # Treat errors as blanks to be safe

def find_best_in_set(image_paths):
    """
    Finds the best image within a given list of image paths.
    """
    best_image = None
    best_score = -1
    for image_path in image_paths:
        properties = analyze_image(image_path)
        if properties:
            score = properties['brightness'] + properties['contrast'] + properties['sharpness']
            if score > best_score:
                best_score = score
                best_image = image_path
    return best_image

def process_images(source_dir, dest_dir):
    """
    Processes all images in the source directory, groups them into sets,
    finds the best image from each set, and copies it to the destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not all_files:
        print("No images found in the source directory.")
        return

    sets = []
    current_set = []

    print("Identifying image sets...")
    for image_path in all_files:
        if is_blank(image_path):
            print(f"'{os.path.basename(image_path)}' is blank. Starting a new set.")
            if current_set:
                sets.append(current_set)
            current_set = [] # Start a new set, discard the blank
        else:
            current_set.append(image_path)
    
    if current_set:
        sets.append(current_set)

    print(f"Found {len(sets)} sets of images.")

    for i, image_set in enumerate(sets):
        print(f"Processing Set {i+1} with {len(image_set)} images...")
        if not image_set:
            print("  Set is empty, skipping.")
            continue
        best_image = find_best_in_set(image_set)
        if best_image:
            print(f"  Best image: '{os.path.basename(best_image)}'")
            shutil.copy(best_image, dest_dir)
            print(f"  Copied to '{dest_dir}'")
        else:
            print("  Could not determine the best image for this set.")

if __name__ == '__main__':
    source_directory = 'D:\\Memex small test'
    keeper_directory = os.path.join(os.path.dirname(source_directory), 'keeper')
    process_images(source_directory, keeper_directory)
