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
    """Analyzes for smiles, open eyes, teeth, and face size."""
    try:
        img = cv2.imread(image_path)
        if img is None: return {'eyes_open': 0, 'smiles': 0, 'teeth_showing': 0, 'face_size': 0}
        
        img_height, img_width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0: return {'eyes_open': 0, 'smiles': 0, 'teeth_showing': 0, 'face_size': 0}
        
        (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
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
            
            if white_percentage > 1.0: # Using the tested 1% threshold
                teeth_showing = 1

        face_size_percentage = (w / img_width) * 100
        
        return {'eyes_open': 1 if len(eyes) >= 2 else 0, 'smiles': has_smile, 'teeth_showing': teeth_showing, 'face_size': face_size_percentage}
    except Exception:
        return {'eyes_open': 0, 'smiles': 0, 'teeth_showing': 0, 'face_size': 0}

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
        self.geometry("600x450")
        self.grid_columnconfigure(0, weight=1)

        # --- Directory Selection ---
        self.directory_frame = ctk.CTkFrame(self)
        self.directory_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.directory_label = ctk.CTkLabel(self.directory_frame, text="No Directory Selected")
        self.directory_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.browse_button = ctk.CTkButton(self.directory_frame, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=1, padx=10, pady=10, sticky="e")

        # --- Threshold Slider ---
        self.slider_frame = ctk.CTkFrame(self)
        self.slider_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.slider_frame.grid_columnconfigure(1, weight=1)
        
        self.headsize_label = ctk.CTkLabel(self.slider_frame, text="Close-up Threshold:")
        self.headsize_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.headsize_slider = ctk.CTkSlider(self.slider_frame, from_=10, to=50, number_of_steps=40, command=self.update_headsize_label)
        self.headsize_slider.set(25)
        self.headsize_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.headsize_value_label = ctk.CTkLabel(self.slider_frame, text="25%")
        self.headsize_value_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        # --- Controls and Progress ---
        self.process_button = ctk.CTkButton(self, text="Grade and Select Images", command=self.start_processing)
        self.process_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.status_label = ctk.CTkLabel(self, text="Select a directory and click 'Process'")
        self.status_label.grid(row=4, column=0, padx=20, pady=10)
        self.time_label = ctk.CTkLabel(self, text="")
        self.time_label.grid(row=5, column=0, padx=20, pady=10)

        self.source_directory = ""
        self.close_up_threshold = 25.0

    def update_headsize_label(self, value):
        self.close_up_threshold = value
        self.headsize_value_label.configure(text=f"{int(value)}%")

    def browse_directory(self):
        self.source_directory = filedialog.askdirectory()
        self.directory_label.configure(text=self.source_directory if self.source_directory else "No Directory Selected")

    def start_processing(self):
        if not self.source_directory:
            self.status_label.configure(text="Please select a directory first.")
            return
        self.process_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.headsize_slider.configure(state="disabled")
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def process_completed_set(self, set_data, dest_dir):
        """Finds the best of each shot type in a set and copies them."""
        best_closeup, best_longshot = None, None
        best_closeup_score, best_longshot_score = -1, -1

        for image_path, total_score, is_closeup in set_data:
            if is_closeup:
                if total_score > best_closeup_score or (total_score == best_closeup_score and (best_closeup is None or os.path.basename(image_path) > os.path.basename(best_closeup))):
                    best_closeup_score = total_score
                    best_closeup = image_path
            else:
                if total_score > best_longshot_score or (total_score == best_longshot_score and (best_longshot is None or os.path.basename(image_path) > os.path.basename(best_longshot))):
                    best_longshot_score = total_score
                    best_longshot = image_path
        
        if best_closeup: shutil.copy(best_closeup, dest_dir)
        if best_longshot: shutil.copy(best_longshot, dest_dir)

    def process_images_thread(self):
        source_dir = self.source_directory
        dest_dir = os.path.join(source_dir, "keepers_final")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_files = len(all_files)
        start_time = time.time()

        current_set_data = []
        for i, image_path in enumerate(all_files):
            self.status_label.configure(text=f"Processing {i + 1}/{total_files}: {os.path.basename(image_path)}")
            
            if is_blank(image_path):
                if current_set_data:
                    self.process_completed_set(current_set_data, dest_dir)
                current_set_data = []
            else:
                facial_features = analyze_facial_features(image_path)
                quality_metrics = analyze_image_quality(image_path)

                if quality_metrics and facial_features['face_size'] > 0:
                    smile_score = facial_features['smiles'] * 40
                    eyes_score = facial_features['eyes_open'] * 20
                    teeth_score = facial_features['teeth_showing'] * 5
                    sharpness_score = min(quality_metrics['sharpness'] / 100.0, 15)
                    contrast_score = min(quality_metrics['contrast'] / 10.0, 15)
                    brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
                    quality_score = sharpness_score + contrast_score + brightness_score
                    total_score = smile_score + eyes_score + teeth_score + quality_score
                    
                    write_score_to_exif(image_path, total_score)
                    is_closeup = facial_features['face_size'] > self.close_up_threshold
                    current_set_data.append((image_path, total_score, is_closeup))

            progress = (i + 1) / total_files
            self.progress_bar.set(progress)
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / (i + 1) if (i + 1) > 0 else 0
            remaining_files = total_files - (i + 1)
            time_remaining = time_per_file * remaining_files
            self.time_label.configure(text=f"Time Remaining: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")

        if current_set_data:
            self.process_completed_set(current_set_data, dest_dir)

        self.progress_bar.set(1.0)
        self.status_label.configure(text=f"Processing Complete. Best images saved to '{dest_dir}'")
        self.time_label.configure(text="")
        self.process_button.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.headsize_slider.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
