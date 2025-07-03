import os
import shutil
import cv2
from PIL import Image
import numpy as np

# --- Haar Cascade Setup ---
HAAR_CASCADE_PATH = cv2.data.haarcascades
try:
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
    EYE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_eye.xml'))
    SMILE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_smile.xml'))
except Exception as e:
    print(f"Error loading Haar Cascade files: {e}")
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
    """
    Analyzes facial features and returns face size, eye status, and smile status.
    Returns face_size as a percentage of image width.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'face_size': 0, 'eyes_open': 0, 'smiles': 0}

        img_height, img_width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'face_size': 0, 'eyes_open': 0, 'smiles': 0}

        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        
        has_open_eyes = 1 if len(eyes) >= 2 else 0
        has_smile = 1 if len(smiles) > 0 else 0
        face_size_percentage = (w / img_width) * 100
        
        return {'face_size': face_size_percentage, 'eyes_open': has_open_eyes, 'smiles': has_smile}

    except Exception as e:
        print(f"Error analyzing facial features of {os.path.basename(image_path)}: {e}")
        return {'face_size': 0, 'eyes_open': 0, 'smiles': 0}

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

def find_best_shots_in_set(image_paths, close_up_threshold=20.0):
    """
    Finds the best close-up and best long-shot from a set.
    `close_up_threshold` is the face width percentage to be considered a close-up.
    """
    best_closeup = None
    best_longshot = None
    best_closeup_score = -1
    best_longshot_score = -1

    print(f"  Using threshold: Face width > {close_up_threshold}% of image width for close-ups.")

    for image_path in image_paths:
        facial_features = analyze_facial_features(image_path)
        quality_metrics = analyze_image_quality(image_path)

        if quality_metrics and facial_features['face_size'] > 0:
            feature_score = (facial_features['smiles'] * 1000) + (facial_features['eyes_open'] * 1000)
            quality_score = quality_metrics['sharpness'] + quality_metrics['contrast']
            total_score = feature_score + quality_score
            
            is_closeup = facial_features['face_size'] > close_up_threshold
            shot_type = "Close-up" if is_closeup else "Long-shot"

            print(f"  - {os.path.basename(image_path)}: Type={shot_type}, FaceSize={facial_features['face_size']:.2f}%, Score={total_score:.2f}")

            if is_closeup:
                if total_score > best_closeup_score:
                    best_closeup_score = total_score
                    best_closeup = image_path
            else:
                if total_score > best_longshot_score:
                    best_longshot_score = total_score
                    best_longshot = image_path
                
    return best_closeup, best_longshot

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
            
        best_closeup, best_longshot = find_best_shots_in_set(image_set)
        
        if best_closeup:
            print(f"  Best Close-up for Set {i+1}: '{os.path.basename(best_closeup)}'")
            shutil.copy(best_closeup, dest_dir)
            print(f"  Copied to '{dest_dir}'")
        else:
            print(f"  No suitable close-up found for Set {i+1}.")

        if best_longshot:
            print(f"  Best Long-shot for Set {i+1}: '{os.path.basename(best_longshot)}'")
            shutil.copy(best_longshot, dest_dir)
            print(f"  Copied to '{dest_dir}'\n")
        else:
            print(f"  No suitable long-shot found for Set {i+1}.\n")

if __name__ == '__main__':
    source_directory = 'D:\\Memex small test'
    keeper_directory = os.path.join(os.path.dirname(source_directory), 'keeper_v3')
    process_images(source_directory, keeper_directory)
