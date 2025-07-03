import os
import cv2
import numpy as np

# --- Haar Cascade Setup ---
HAAR_CASCADE_PATH = cv2.data.haarcascades
try:
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
except Exception as e:
    print(f"Error loading Haar Cascade files: {e}")
    exit()

def analyze_face_size(image_path):
    """Analyzes an image to find the size of the largest face."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  - Could not read image.")
            return None

        img_height, img_width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use slightly adjusted parameters for potentially better detection
        faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
        
        if len(faces) == 0:
            return 0 # No face detected

        # Find the largest face
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        
        # Calculate face width as a percentage of image width
        face_size_percentage = (w / img_width) * 100
        
        return face_size_percentage

    except Exception as e:
        print(f"  - Error analyzing {os.path.basename(image_path)}: {e}")
        return None

def run_test(source_dir, close_up_threshold=20.0):
    """Runs the face size classification test."""
    print(f"--- Running Face Size Classification Test ---")
    print(f"Directory: {source_dir}")
    print(f"Close-up Threshold: > {close_up_threshold}%")
    print("-" * 40)

    all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not all_files:
        print("No images found in the directory.")
        return

    for image_path in all_files:
        print(f"Analyzing '{os.path.basename(image_path)}'...")
        face_size = analyze_face_size(image_path)

        if face_size is None:
            continue
        
        if face_size == 0:
            classification = "No Face Detected"
        elif face_size > close_up_threshold:
            classification = "Close-up"
        else:
            classification = "Long-shot"
            
        print(f"  - Detected Face Size: {face_size:.2f}% of image width")
        print(f"  - Classification: {classification}")
        print("-" * 20)

if __name__ == "__main__":
    test_directory = "D:\\Memex close long"
    run_test(test_directory)
