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
import subprocess
import mediapipe as mp

# --- MediaPipe and Haar Cascade Setup ---
mp_face_mesh = mp.solutions.face_mesh
HAAR_CASCADE_PATH = cv2.data.haarcascades
try:
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
except Exception as e:
    print(f"Error loading Haar Cascade files: {e}")
    exit()

# --- Landmark Indices ---
LEFT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDXS = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

def calculate_ear(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR)."""
    p2_p6 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4) if p1_p4 > 0 else 0.0

def analyze_facial_features(image_path):
    """Analyzes facial features using MediaPipe."""
    try:
        img = cv2.imread(image_path)
        if img is None: return {'eyes_open': 0, 'smile_score': 0, 'face_size': 0, 'head_turn_score': 0, 'teeth_showing': 0}
        
        img_height, img_width, _ = img.shape
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return {'eyes_open': 0, 'smile_score': 0, 'face_size': 0, 'head_turn_score': 0, 'teeth_showing': 0}

            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Eye Openness (EAR) ---
            left_eye_pts = [landmarks[i] for i in LEFT_EYE_EAR_IDXS]
            right_eye_pts = [landmarks[i] for i in RIGHT_EYE_EAR_IDXS]
            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            eyes_open = 1 if avg_ear > 0.2 else 0
            
            # --- Smile Score (MAR) ---
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            vertical_dist = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
            horizontal_dist = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))
            mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
            smile_score = min(max(mar * 400 - 10, 0), 40)

            # --- Head Turn Score (Symmetry Ratio) ---
            nose_tip = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
            left_eye_center = (np.array([landmarks[LEFT_EYE_CORNERS[0]].x, landmarks[LEFT_EYE_CORNERS[0]].y]) + np.array([landmarks[LEFT_EYE_CORNERS[1]].x, landmarks[LEFT_EYE_CORNERS[1]].y])) / 2.0
            right_eye_center = (np.array([landmarks[RIGHT_EYE_CORNERS[0]].x, landmarks[RIGHT_EYE_CORNERS[0]].y]) + np.array([landmarks[RIGHT_EYE_CORNERS[1]].x, landmarks[RIGHT_EYE_CORNERS[1]].y])) / 2.0
            dist_a = np.linalg.norm(nose_tip - left_eye_center)
            dist_b = np.linalg.norm(nose_tip - right_eye_center)
            symmetry_ratio = dist_a / dist_b if dist_b > 0 else 0
            deviation = abs(1.0 - symmetry_ratio)
            head_turn_score = max(0, 20 - (deviation / 0.2) * 20)

            # --- Teeth Detection (Haar Cascade) ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            teeth_showing = 0
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
                # This part is simplified as MediaPipe already gives us the face.
                # A full implementation would use the landmarks to define the mouth ROI.
                # For now, we assume the Haar method is sufficient for a bonus.

            # --- Face Size ---
            face_landmarks_x = [lm.x * img_width for lm in landmarks]
            face_landmarks_y = [lm.y * img_height for lm in landmarks]
            face_w = max(face_landmarks_x) - min(face_landmarks_x)
            face_size_percentage = (face_w / img_width) * 100
            
            return {'eyes_open': eyes_open, 'smile_score': smile_score, 'face_size': face_size_percentage, 'head_turn_score': head_turn_score, 'teeth_showing': 0} # Teeth showing logic needs full implementation
            
    except Exception:
        return {'eyes_open': 0, 'smile_score': 0, 'face_size': 0, 'head_turn_score': 0, 'teeth_showing': 0}

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

def write_grade_to_exif(image_path, score):
    """Writes the grade to the image's EXIF data using ExifTool."""
    try:
        command = ['exiftool', '-overwrite_original', f'-ImageDescription=PhotoGrade: {score:.2f}', image_path]
        subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Could not write grade to {os.path.basename(image_path)}: {e}")

def add_5_star_rating_exiftool(image_path):
    """Adds a 5-star rating using ExifTool."""
    try:
        command = ['exiftool', '-overwrite_original', '-Rating=5', image_path]
        subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Could not write 5-star rating to {os.path.basename(image_path)}: {e}")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Grader and Selector")
        self.geometry("600x600")
        self.grid_columnconfigure(0, weight=1)

        # --- UI Elements ---
        self.source_dir_frame = ctk.CTkFrame(self)
        self.source_dir_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.source_dir_label = ctk.CTkLabel(self.source_dir_frame, text="Source Directory:")
        self.source_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.source_dir_path_label = ctk.CTkLabel(self.source_dir_frame, text="Not Selected")
        self.source_dir_path_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.source_dir_browse_button = ctk.CTkButton(self.source_dir_frame, text="Browse", command=self.browse_source_directory)
        self.source_dir_browse_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.source_dir_frame.grid_columnconfigure(1, weight=1)

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

        self.process_button = ctk.CTkButton(self, text="Grade and Select Images", command=self.start_processing)
        self.process_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.target_dir_frame = ctk.CTkFrame(self)
        self.target_dir_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.target_dir_label = ctk.CTkLabel(self.target_dir_frame, text="Target Directory for Ratings:")
        self.target_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.target_dir_path_label = ctk.CTkLabel(self.target_dir_frame, text="Not Selected")
        self.target_dir_path_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.target_dir_browse_button = ctk.CTkButton(self.target_dir_frame, text="Browse", command=self.browse_target_directory)
        self.target_dir_browse_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.target_dir_frame.grid_columnconfigure(1, weight=1)

        self.transfer_button = ctk.CTkButton(self, text="Transfer 5-Star Ratings to Target", command=self.start_transfer, state="disabled")
        self.transfer_button.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="Select a source directory and click 'Grade'")
        self.status_label.grid(row=6, column=0, padx=20, pady=10)
        self.time_label = ctk.CTkLabel(self, text="")
        self.time_label.grid(row=7, column=0, padx=20, pady=10)

        self.source_directory = ""
        self.target_directory = ""
        self.close_up_threshold = 25.0
        self.keeper_dir = ""

    def update_headsize_label(self, value):
        self.close_up_threshold = value
        self.headsize_value_label.configure(text=f"{int(value)}%")

    def browse_source_directory(self):
        self.source_directory = filedialog.askdirectory()
        self.source_dir_path_label.configure(text=self.source_directory if self.source_directory else "Not Selected")

    def browse_target_directory(self):
        self.target_directory = filedialog.askdirectory()
        self.target_dir_path_label.configure(text=self.target_directory if self.target_directory else "Not Selected")

    def start_processing(self):
        if not self.source_directory:
            self.status_label.configure(text="Please select a source directory first.")
            return
        self.set_ui_state("disabled")
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def start_transfer(self):
        if not self.target_directory:
            self.status_label.configure(text="Please select a target directory first.")
            return
        if not self.keeper_dir or not os.path.exists(self.keeper_dir):
            self.status_label.configure(text="Please run the grading process first to generate a keeper folder.")
            return
        self.set_ui_state("disabled")
        threading.Thread(target=self.transfer_ratings_thread, daemon=True).start()

    def set_ui_state(self, state):
        """Enable or disable all interactive UI elements."""
        is_normal = state == "normal"
        self.process_button.configure(state=state)
        self.source_dir_browse_button.configure(state=state)
        self.headsize_slider.configure(state=state)
        self.target_dir_browse_button.configure(state=state)
        self.transfer_button.configure(state="normal" if is_normal and self.keeper_dir else "disabled")

    def process_completed_set(self, set_data, dest_dir):
        """Finds the best of each shot type and copies them."""
        best_closeup, best_longshot = None, None
        best_closeup_score, best_longshot_score = -1, -1

        for image_path, total_score, is_closeup, eyes_open in set_data:
            if not eyes_open:
                continue

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
        self.keeper_dir = os.path.join(self.source_directory, "keepers_rated")
        if os.path.exists(self.keeper_dir):
            shutil.rmtree(self.keeper_dir)
        os.makedirs(self.keeper_dir)

        all_files = sorted([os.path.join(self.source_directory, f) for f in os.listdir(self.source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_files = len(all_files)
        start_time = time.time()

        current_set_data = []
        for i, image_path in enumerate(all_files):
            self.status_label.configure(text=f"Processing {i + 1}/{total_files}: {os.path.basename(image_path)}")
            
            if is_blank(image_path):
                if current_set_data:
                    self.process_completed_set(current_set_data, self.keeper_dir)
                current_set_data = []
            else:
                facial_features = analyze_facial_features(image_path)
                quality_metrics = analyze_image_quality(image_path)

                if quality_metrics and facial_features['face_size'] > 0:
                    # --- NEW SCORING MODEL ---
                    eyes_score = facial_features['eyes_open'] * 20
                    
                    # Quality Score (up to 30 points)
                    sharpness_score = min(quality_metrics['sharpness'] / 133.0, 10)
                    contrast_score = min(quality_metrics['contrast'] / 10.0, 10)
                    brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
                    quality_score = sharpness_score + contrast_score + brightness_score
                    
                    # Total Score
                    total_score = facial_features['smile_score'] + facial_features['head_turn_score'] + eyes_score + quality_score + (facial_features['teeth_showing'] * 5)
                    
                    write_grade_to_exif(image_path, total_score)
                    is_closeup = facial_features['face_size'] > self.close_up_threshold
                    current_set_data.append((image_path, total_score, is_closeup, facial_features['eyes_open']))

            progress = (i + 1) / total_files
            self.progress_bar.set(progress)
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / (i + 1) if (i + 1) > 0 else 0
            remaining_files = total_files - (i + 1)
            time_remaining = time_per_file * remaining_files
            self.time_label.configure(text=f"Time Remaining: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")

        if current_set_data:
            self.process_completed_set(current_set_data, self.keeper_dir)

        # --- Final Step: Rate all keepers ---
        self.status_label.configure(text="Rating keeper images...")
        keeper_files = [os.path.join(self.keeper_dir, f) for f in os.listdir(self.keeper_dir)]
        for keeper_path in keeper_files:
            add_5_star_rating_exiftool(keeper_path)

        self.progress_bar.set(1.0)
        self.status_label.configure(text=f"Processing Complete. 5-star keepers saved to '{self.keeper_dir}'")
        self.time_label.configure(text="")
        self.set_ui_state("normal")

    def transfer_ratings_thread(self):
        self.status_label.configure(text="Starting rating transfer...")
        
        keeper_files = [f for f in os.listdir(self.keeper_dir)]
        target_files = {os.path.splitext(f)[0]: os.path.join(self.target_directory, f) for f in os.listdir(self.target_directory)}
        
        transferred_count = 0
        for i, keeper_file in enumerate(keeper_files):
            keeper_basename = os.path.splitext(keeper_file)[0]
            self.status_label.configure(text=f"Searching for {keeper_basename} in target directory...")
            
            if keeper_basename in target_files:
                target_path = target_files[keeper_basename]
                self.status_label.configure(text=f"Found. Applying 5-star rating to {os.path.basename(target_path)}...")
                add_5_star_rating_exiftool(target_path)
                transferred_count += 1
            
            self.progress_bar.set((i + 1) / len(keeper_files))

        self.status_label.configure(text=f"Transfer Complete. {transferred_count} ratings applied.")
        self.set_ui_state("normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()
