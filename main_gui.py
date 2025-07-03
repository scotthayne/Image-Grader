import os
import shutil
import cv2
import piexif
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import threading

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
            sharpness = cv2.Laplacian(grayscale_array, cv2.CV_64F).var()
            return {'sharpness': sharpness}
    except Exception:
        return None

def analyze_facial_features(image_path):
    """Analyzes facial features (eyes open, smiling)."""
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
    """Writes the calculated score to the image's EXIF data."""
    try:
        exif_dict = piexif.load(image_path)
        exif_dict['0th'][piexif.ImageIFD.ImageDescription] = f"PhotoGrade: {score:.2f}".encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
    except Exception as e:
        print(f"Could not write EXIF data to {os.path.basename(image_path)}: {e}")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Set Best Selector")
        self.geometry("500x350")

        self.grid_columnconfigure(0, weight=1)

        self.directory_frame = ctk.CTkFrame(self)
        self.directory_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.directory_label = ctk.CTkLabel(self.directory_frame, text="No Directory Selected")
        self.directory_label.grid(row=0, column=0, padx=10, pady=10)

        self.browse_button = ctk.CTkButton(self.directory_frame, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=1, padx=10, pady=10)

        self.process_button = ctk.CTkButton(self, text="Process Images", command=self.start_processing)
        self.process_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.grid(row=3, column=0, padx=20, pady=10)

        self.source_directory = ""

    def browse_directory(self):
        self.source_directory = filedialog.askdirectory()
        self.directory_label.configure(text=self.source_directory if self.source_directory else "No Directory Selected")

    def start_processing(self):
        if not self.source_directory:
            self.status_label.configure(text="Please select a directory first.")
            return
        
        self.process_button.configure(state="disabled")
        self.status_label.configure(text="Processing...")
        
        # Run processing in a separate thread to keep the GUI responsive
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def process_images_thread(self):
        source_dir = self.source_directory
        dest_dir = os.path.join(source_dir, "keepers")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        all_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(all_files)

        for i, image_path in enumerate(all_files):
            facial_features = analyze_facial_features(image_path)
            quality_metrics = analyze_image_quality(image_path)

            if quality_metrics:
                # Weighted scoring
                smile_score = facial_features['smiles'] * 50
                eyes_score = facial_features['eyes_open'] * 25
                # Normalize sharpness to a 0-25 scale (assuming max sharpness around 2000)
                quality_score = min(quality_metrics['sharpness'] / 80, 25)
                
                total_score = smile_score + eyes_score + quality_score
                
                write_score_to_exif(image_path, total_score)

                # Simple logic to copy high-scoring photos to keepers
                if total_score > 75:
                    shutil.copy(image_path, dest_dir)

            # Update GUI
            progress = (i + 1) / total_files
            self.progress_bar.set(progress)
            self.status_label.configure(text=f"Processing {i+1}/{total_files}")

        self.status_label.configure(text=f"Processing Complete. Keepers saved to '{dest_dir}'")
        self.process_button.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
