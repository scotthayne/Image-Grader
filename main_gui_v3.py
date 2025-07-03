import os
import shutil
import cv2
import piexif
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import threading
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

def is_blank(image_path):
    """Checks if an image is blank (contains no faces)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return True
        faces = FACE_CASCADE.detectMultiScale(img, 1.1, 4)
        return len(faces) == 0
    except Exception:
        return True

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
    except Exception:
        try:
            zeroth_ifd = {piexif.ImageIFD.ImageDescription: f"PhotoGrade: {score:.2f}".encode('utf-8')}
            exif_dict = {"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
        except Exception as e:
            print(f"Could not write EXIF data to {os.path.basename(image_path)}: {e}")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Grader and Selector")
        self.geometry("600x400")
        self.grid_columnconfigure(0, weight=1)

        self.directory_frame = ctk.CTkFrame(self)
        self.directory_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.directory_label = ctk.CTkLabel(self.directory_frame, text="No Directory Selected")
        self.directory_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.browse_button = ctk.CTkButton(self.directory_frame, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=1, padx=10, pady=10, sticky="e")

        self.process_button = ctk.CTkButton(self, text="Grade and Select Images", command=self.start_processing)
        self.process_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="Select a directory and click 'Process'")
        self.status_label.grid(row=3, column=0, padx=20, pady=10)
        
        self.time_label = ctk.CTkLabel(self, text="")
        self.time_label.grid(row=4, column=0, padx=20, pady=10)

        self.source_directory = ""

    def browse_directory(self):
        self.source_directory = filedialog.askdirectory()
        self.directory_label.configure(text=self.source_directory if self.source_directory else "No Directory Selected")

    def start_processing(self):
        if not self.source_directory:
            self.status_label.configure(text="Please select a directory first.")
            return
        self.process_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def process_images_thread(self):
        source_dir = self.source_directory
        dest_dir = os.path.join(source_dir, "keepers_best_in_set")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_files = len(all_files)
        start_time = time.time()

        # Stage 1: Identify sets
        self.status_label.configure(text="Identifying image sets...")
        sets = []
        current_set = []
        for i, image_path in enumerate(all_files):
            self.progress_bar.set((i + 1) / total_files)
            if is_blank(image_path):
                if current_set:
                    sets.append(current_set)
                current_set = []
            else:
                current_set.append(image_path)
        if current_set:
            sets.append(current_set)
        
        self.status_label.configure(text=f"Found {len(sets)} sets. Now grading images...")
        time.sleep(2) # Pause to show the message

        # Stage 2: Process each set
        processed_files = 0
        for set_index, image_set in enumerate(sets):
            best_image_in_set = None
            best_score_in_set = -1

            for image_path in image_set:
                self.status_label.configure(text=f"Processing Set {set_index + 1}/{len(sets)}: {os.path.basename(image_path)}")
                
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
                    
                    write_score_to_exif(image_path, total_score)

                    # Check for new best image
                    if total_score > best_score_in_set:
                        best_score_in_set = total_score
                        best_image_in_set = image_path
                    elif total_score == best_score_in_set:
                        # Tie-breaker: higher filename wins
                        if best_image_in_set is None or os.path.basename(image_path) > os.path.basename(best_image_in_set):
                            best_image_in_set = image_path
                
                processed_files += 1
                progress = processed_files / total_files
                self.progress_bar.set(progress)
                
                elapsed_time = time.time() - start_time
                time_per_file = elapsed_time / processed_files
                remaining_files = total_files - processed_files
                time_remaining = time_per_file * remaining_files
                self.time_label.configure(text=f"Time Remaining: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")

            if best_image_in_set:
                shutil.copy(best_image_in_set, dest_dir)

        self.status_label.configure(text=f"Processing Complete. Best images saved to '{dest_dir}'")
        self.time_label.configure(text="")
        self.process_button.configure(state="normal")
        self.browse_button.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
