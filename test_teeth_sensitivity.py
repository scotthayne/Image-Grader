import os
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

def get_white_percentage_in_smile(image_path):
    """
    Analyzes an image to find the percentage of white pixels in a detected smile.
    Returns the percentage if a smile is found, otherwise None.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None # No face
        
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        
        if len(smiles) == 0:
            return 0 # No smile

        (sx, sy, sw, sh) = smiles[0]
        smile_roi = roi_color[sy:sy+sh, sx:sx+sw]
        hsv_smile = cv2.cvtColor(smile_roi, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv_smile, lower_white, upper_white)
        white_percentage = (cv2.countNonZero(mask) / (sw * sh)) * 100 if (sw * sh) > 0 else 0
        
        return white_percentage
    except Exception as e:
        print(f"  - Error during analysis: {e}")
        return None

def run_sensitivity_test(source_dir):
    """Runs the teeth detection sensitivity test."""
    print(f"--- Running Teeth Detection Sensitivity Test ---")
    print(f"Directory: {source_dir}")
    print("-" * 50)

    all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not all_files:
        print("No images found in the directory.")
        return

    for image_path in all_files:
        print(f"Analyzing '{os.path.basename(image_path)}'...")
        white_percentage = get_white_percentage_in_smile(image_path)

        if white_percentage is None:
            print("  - No face detected or error during analysis.")
        elif white_percentage == 0:
            print("  - No smile detected.")
        else:
            print(f"  - Detected white pixel percentage in smile: {white_percentage:.2f}%")
            print("  - Detection results at different sensitivity thresholds:")
            for sensitivity in range(1, 11):
                detected = "Yes" if white_percentage > sensitivity else "No"
                print(f"    - At > {sensitivity}% sensitivity: Teeth Detected = {detected}")
        print("-" * 50)

if __name__ == "__main__":
    test_directory = "D:\\Memex teeth"
    run_sensitivity_test(test_directory)
