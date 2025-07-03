import os
import shutil
import cv2
from PIL import Image
import numpy as np

# --- Haar Cascade Setup ---
# OpenCV provides a path to its data files, which include the Haar cascades.
HAAR_CASCADE_PATH = cv2.data.haarcascades

# Load the pre-trained Haar Cascade classifiers
try:
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
    EYE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_eye.xml'))
    SMILE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_smile.xml'))
except Exception as e:
    print(f"Error loading Haar Cascade files: {e}")
    print("Please ensure OpenCV is installed correctly.")
    exit()

def analyze_image_quality(image_path):
    """Analyzes basic image quality (brightness, contrast, sharpness)."""
    try:
        with Image.open(image_path) as img:
            grayscale_img = img.convert('L')
            grayscale_array = np.array(grayscale_img)
            brightness = np.mean(grayscale_array)
            contrast = np.std(grayscale_array)
            laplacian = cv2.Laplacian(grayscale_array, cv2.CV_64F)
            sharpness = laplacian.var()
            return {'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness}
    except Exception as e:
        print(f"Error analyzing quality of {os.path.basename(image_path)}: {e}")
        return None

def analyze_facial_features(image_path):
    """Analyzes facial features (eyes open, smiling)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'eyes_open': 0, 'smiles': 0}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'eyes_open': 0, 'smiles': 0} # No faces detected

        # We only analyze the largest detected face
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        
        # Detect smiles within the face ROI
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        
        # A simple proxy for eyes being open is detecting two eyes.
        has_open_eyes = 1 if len(eyes) >= 2 else 0
        has_smile = 1 if len(smiles) > 0 else 0
        
        return {'eyes_open': has_open_eyes, 'smiles': has_smile}

    except Exception as e:
        print(f"Error analyzing facial features of {os.path.basename(image_path)}: {e}")
        return {'eyes_open': 0, 'smiles': 0}

def is_blank(image_path):
    """Checks if an image is blank (contains no faces)."""
    try:
        img = cv2.imread(image_path)
        if img is None: return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        return len(faces) == 0
    except Exception:
        return True

def find_best_in_set(image_paths):
    """Finds the best image in a set based on a combined scoring function."""
    best_image = None
    best_score = -1

    for image_path in image_paths:
        facial_features = analyze_facial_features(image_path)
        quality_metrics = analyze_image_quality(image_path)

        if quality_metrics:
            # Primary score from facial features (high weight)
            # A perfect score here is 2000 (1000 for smile, 1000 for eyes)
            feature_score = (facial_features['smiles'] * 1000) + (facial_features['eyes_open'] * 1000)
            
            # Secondary score from image quality (lower weight, acts as tie-breaker)
            quality_score = quality_metrics['sharpness'] + quality_metrics['contrast']
            
            total_score = feature_score + quality_score
            
            print(f"  - {os.path.basename(image_path)}: Score={total_score:.2f} (Features={feature_score}, Quality={quality_score:.2f})")

            if total_score > best_score:
                best_score = total_score
                best_image = image_path
                
    return best_image

def process_images(source_dir, dest_dir):
    """Processes all images, finds the best from each set, and copies it."""
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
            current_set = []
        else:
            current_set.append(image_path)
    
    if current_set:
        sets.append(current_set)

    print(f"\nFound {len(sets)} sets of images.\n")

    for i, image_set in enumerate(sets):
        print(f"--- Processing Set {i+1} with {len(image_set)} images ---")
        if not image_set:
            print("  Set is empty, skipping.")
            continue
        best_image = find_best_in_set(image_set)
        if best_image:
            print(f"  Best image for Set {i+1}: '{os.path.basename(best_image)}'")
            shutil.copy(best_image, dest_dir)
            print(f"  Copied to '{dest_dir}'\n")
        else:
            print("  Could not determine the best image for this set.\n")

if __name__ == '__main__':
    source_directory = 'D:\\Memex samples'
    keeper_directory = os.path.join(os.path.dirname(source_directory), 'keeper_v2')
    process_images(source_directory, keeper_directory)
