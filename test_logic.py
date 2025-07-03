import os
import shutil
import cv2
import piexif
from PIL import Image
import numpy as np
import time

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
    """Analyzes sharpness, contrast, and brightness."""
    try:
        with Image.open(image_path) as img:
            grayscale_img = img.convert('L')
            grayscale_array = np.array(grayscale_img)
            
            sharpness = cv2.Laplacian(grayscale_array, cv2.CV_64F).var()
            contrast = grayscale_array.std()
            brightness = grayscale_array.mean()
            
            return {'sharpness': sharpness, 'contrast': contrast, 'brightness': brightness}
    except Exception:
        return None

def analyze_facial_features(image_path):
    """Analyzes for smiles and open eyes."""
    try:
        img = cv2.imread(image_path)
        if img is None: return {'eyes_open': 0, 'smiles': 0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return {'eyes_open': 0, 'smiles': 0}
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        return {'eyes_open': 1 if len(eyes) >= 2 else 0, 'smiles': 1 if len(smiles) > 0 else 0}
    except Exception:
        return {'eyes_open': 0, 'smiles': 0}

def write_score_to_exif(image_path, score):
    """Writes the score to the image's EXIF data."""
    try:
        exif_dict = piexif.load(image_path)
        exif_dict['0th'][piexif.ImageIFD.ImageDescription] = f"PhotoGrade: {score:.2f}".encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        print(f"  - Wrote EXIF score {score:.2f} to {os.path.basename(image_path)}")
    except Exception:
        try:
            zeroth_ifd = {piexif.ImageIFD.ImageDescription: f"PhotoGrade: {score:.2f}".encode('utf-8')}
            exif_dict = {"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            print(f"  - Wrote EXIF score {score:.2f} to {os.path.basename(image_path)} (new EXIF data created)")
        except Exception as e:
            print(f"  - Could not write EXIF data to {os.path.basename(image_path)}: {e}")

def process_images(source_dir):
    """The core processing logic, adapted for command-line execution."""
    dest_dir = os.path.join(source_dir, "keepers_graded_test")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(all_files)} files in '{source_dir}'...")

    for image_path in all_files:
        print(f"\nAnalyzing {os.path.basename(image_path)}...")
        facial_features = analyze_facial_features(image_path)
        quality_metrics = analyze_image_quality(image_path)

        if quality_metrics:
            smile_score = facial_features['smiles'] * 40
            eyes_score = facial_features['eyes_open'] * 20
            
            sharpness_score = min(quality_metrics['sharpness'] / 100.0, 15)
            contrast_score = min(quality_metrics['contrast'] / 10.0, 15)
            brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
            
            quality_score = sharpness_score + contrast_score + brightness_score
            total_score = smile_score + eyes_score + quality_score
            
            print(f"  - Facial Features: Smile={facial_features['smiles']}, Eyes Open={facial_features['eyes_open']}")
            print(f"  - Quality Metrics: Sharpness={quality_metrics['sharpness']:.2f}, Contrast={quality_metrics['contrast']:.2f}, Brightness={quality_metrics['brightness']:.2f}")
            print(f"  - Calculated Score: {total_score:.2f}")
            
            write_score_to_exif(image_path, total_score)

            if total_score > 80:
                print(f"  - Score > 80. Copying to '{dest_dir}'")
                shutil.copy(image_path, dest_dir)
        else:
            print("  - Could not analyze image quality.")

    print(f"\nProcessing Complete. Test keepers saved to '{dest_dir}'")
    return dest_dir

if __name__ == "__main__":
    test_directory = "D:\\Memex small test"
    process_images(test_directory)
