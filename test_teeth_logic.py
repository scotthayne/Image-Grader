import os
import shutil
import cv2
import numpy as np

# --- Haar Cascade Setup ---
HAAR_CASCADE_PATH = cv2.data.haarcascades
try:
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
    SMILE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_smile.xml'))
except Exception as e:
    print(f"Error loading Haar Cascade files: {e}")
    exit()

def analyze_facial_features(image_path):
    """Analyzes for smiles and teeth."""
    try:
        img = cv2.imread(image_path)
        if img is None: return {'smiles': 0, 'teeth_showing': 0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0: return {'smiles': 0, 'teeth_showing': 0}
        
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        
        has_smile = 1 if len(smiles) > 0 else 0
        teeth_showing = 0
        if has_smile:
            (sx, sy, sw, sh) = smiles[0]
            smile_roi = roi_color[sy:sy+sh, sx:sx+sw]
            hsv_smile = cv2.cvtColor(smile_roi, cv2.COLOR_BGR2HSV)
            
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            
            mask = cv2.inRange(hsv_smile, lower_white, upper_white)
            white_percentage = (cv2.countNonZero(mask) / (sw * sh)) * 100 if (sw * sh) > 0 else 0
            
            if white_percentage > 5:
                teeth_showing = 1

        return {'smiles': has_smile, 'teeth_showing': teeth_showing}
    except Exception as e:
        print(f"  - Error during facial analysis: {e}")
        return {'smiles': 0, 'teeth_showing': 0}

def run_test(source_dir):
    """Runs the teeth detection logic test."""
    print(f"--- Running Teeth Detection Test ---")
    print(f"Directory: {source_dir}")
    print("-" * 40)

    all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not all_files:
        print("No images found in the directory.")
        return

    for image_path in all_files:
        print(f"Analyzing '{os.path.basename(image_path)}'...")
        features = analyze_facial_features(image_path)

        smile_score = features['smiles'] * 40
        teeth_score = features['teeth_showing'] * 5
        
        print(f"  - Smile Detected: {'Yes' if features['smiles'] else 'No'}")
        print(f"  - Teeth Detected: {'Yes' if features['teeth_showing'] else 'No'}")
        print(f"  - Smile Score: {smile_score}")
        print(f"  - Teeth Bonus: {teeth_score}")
        print("-" * 20)

if __name__ == "__main__":
    test_directory = "D:\\Memex close long"
    run_test(test_directory)
