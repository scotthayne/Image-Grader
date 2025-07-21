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

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
def get_rated_keepers(directory):
    """Scans a directory and returns a list of files with a 5-star rating."""
    rated_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            try:
                command = ['exiftool', '-s', '-s', '-s', '-Rating', image_path]
                result = subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                rating = result.stdout.strip()
                if rating == "5":
                    rated_files.append(image_path)
            except Exception:
                pass
    return rated_files


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
        if img is None: return None
        
        img_height, img_width, _ = img.shape
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye_pts = [landmarks[i] for i in LEFT_EYE_EAR_IDXS]
            right_eye_pts = [landmarks[i] for i in RIGHT_EYE_EAR_IDXS]
            avg_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
            eyes_open = 1 if avg_ear > 0.14 else 0
            
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            vertical_dist = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
            horizontal_dist = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))
            mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
            smile_score = min(max(mar * 500 - 15, 0), 50)
            teeth_bonus = 5 if mar > 0.1 else 0

            nose_tip = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
            left_eye_center = (np.array([landmarks[LEFT_EYE_CORNERS[0]].x, landmarks[LEFT_EYE_CORNERS[0]].y]) + np.array([landmarks[LEFT_EYE_CORNERS[1]].x, landmarks[LEFT_EYE_CORNERS[1]].y])) / 2.0
            right_eye_center = (np.array([landmarks[RIGHT_EYE_CORNERS[0]].x, landmarks[RIGHT_EYE_CORNERS[0]].y]) + np.array([landmarks[RIGHT_EYE_CORNERS[1]].x, landmarks[RIGHT_EYE_CORNERS[1]].y])) / 2.0
            dist_a = np.linalg.norm(nose_tip - left_eye_center)
            dist_b = np.linalg.norm(nose_tip - right_eye_center)
            symmetry_ratio = dist_a / dist_b if dist_b > 0 else 0
            deviation = abs(1.0 - symmetry_ratio)
            head_turn_score = max(0, 20 - (deviation / 0.2) * 20)

            face_landmarks_x = [lm.x * img_width for lm in landmarks]
            face_w = max(face_landmarks_x) - min(face_landmarks_x)
            face_size_percentage = (face_w / img_width) * 100
            
            return {'eyes_open': eyes_open, 'smile_score': smile_score, 'face_size': face_size_percentage, 'head_turn_score': head_turn_score, 'teeth_bonus': teeth_bonus}
            
    except Exception:
        return None

def is_blank(image_path, size_threshold_mb):
    """Determines if an image is blank based on file size and absence of people."""
    try:
        # Check 1: File size less than the threshold
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)  # Convert bytes to MB
        if file_size_mb >= size_threshold_mb:
            return False
        
        # Check 2: No person detected
        facial_features = analyze_facial_features(image_path)
        return facial_features is None  # If no face found, consider it blank
        
    except Exception:
        return True

def detect_person(image_path):
    """Detects if a person is present in the image using HOG detector."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Initialize HOG person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people in the image
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
        
        return len(boxes) > 0  # Return True if any person detected
        
    except Exception:
        return False

def analyze_image_quality(image_path):
    """Analyzes image quality."""
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
    """Writes grade using ExifTool."""
    try:
def add_4_star_rating_exiftool(image_path):
    """Adds 4-star rating using ExifTool."""
    try:
        command = ['exiftool', '-overwrite_original', '-Rating=4', image_path]
        subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Could not write 4-star rating to {os.path.basename(image_path)}: {e}")

        command = ['exiftool', '-overwrite_original', f'-ImageDescription=PhotoGrade: {score:.2f}', image_path]
        subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Could not write grade to {os.path.basename(image_path)}: {e}")

def add_5_star_rating_exiftool(image_path):
    """Adds 5-star rating using ExifTool."""
    try:
        command = ['exiftool', '-overwrite_original', '-Rating=5', image_path]
        subprocess.run(command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Could not write 5-star rating to {os.path.basename(image_path)}: {e}")


class App(ctk.CTk):


    def __init__(self):

        super().__init__()
        self.title("Image Grader and Selector")
        self.geometry("600x650")
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
        self.headsize_slider.set(20) # --- UPDATED DEFAULT ---
        self.headsize_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.headsize_value_label = ctk.CTkLabel(self.slider_frame, text="20%") # --- UPDATED DEFAULT ---
        self.headsize_value_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        self.blank_size_label = ctk.CTkLabel(self.slider_frame, text="Blank Size Threshold (MB):")
        self.blank_size_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.blank_size_slider = ctk.CTkSlider(self.slider_frame, from_=3.0, to=7.0, number_of_steps=40, command=self.update_blank_size_label)
        self.blank_size_slider.set(5.0)
        self.blank_size_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.blank_size_value_label = ctk.CTkLabel(self.slider_frame, text="5.0 MB")
        self.blank_size_value_label.grid(row=1, column=2, padx=10, pady=5, sticky="e")
        
        self.blank_size_desc_label = ctk.CTkLabel(self.slider_frame, text="Set this to the file size of your largest blank separator image.", text_color="gray60")
        self.blank_size_desc_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w")




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
        self.close_up_threshold = 20.0 # --- UPDATED DEFAULT ---
        self.blank_size_threshold = 5.0
        self.processing_complete = False

    def update_headsize_label(self, value):
        self.close_up_threshold = value
        self.headsize_value_label.configure(text=f"{int(value)}%")

    def update_blank_size_label(self, value):
        self.blank_size_threshold = value
        self.blank_size_value_label.configure(text=f"{value:.1f} MB")
        
    def browse_source_directory(self):
        self.source_directory = filedialog.askdirectory()
        self.source_dir_path_label.configure(text=self.source_directory if self.source_directory else "Not Selected")
        if self.source_directory:
            self.process_button.configure(state="normal")  # Re-enable when new directory is selected

    def browse_target_directory(self):
        self.target_directory = filedialog.askdirectory()
        self.target_dir_path_label.configure(text=self.target_directory if self.target_directory else "Not Selected")
        self.set_ui_state("normal")  # Refresh UI state to enable transfer button if ready

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
        if not self.processing_complete:
            self.status_label.configure(text="Please run the grading process first.")
            return
        self.set_ui_state("disabled")
        threading.Thread(target=self.transfer_ratings_thread, daemon=True).start()

    def set_ui_state(self, state):
        """Enable or disable all interactive UI elements."""
        is_normal = state == "normal"
        self.process_button.configure(state=state)
        self.source_dir_browse_button.configure(state=state)
        self.headsize_slider.configure(state=state)
        self.blank_size_slider.configure(state=state)
        self.target_dir_browse_button.configure(state=state)
        self.transfer_button.configure(state="normal" if is_normal and self.target_directory and self.processing_complete else "disabled")

    def process_completed_set(self, set_data):
        """Finds the best images in a set and applies a 5-star rating to close-ups and 4-star to long shots."""
        qualified_images = [img for img in set_data if img[3]]
        
        best_closeup, best_longshot = None, None
        best_closeup_score, best_longshot_score = -1, -1
        
        for image_path, total_score, is_closeup, _ in qualified_images:
            if is_closeup:
                if total_score > best_closeup_score:
                    best_closeup_score = total_score
                    best_closeup = image_path
            else:
                if total_score > best_longshot_score:
                    best_longshot_score = total_score
                    best_longshot = image_path
        
        # Apply ratings based on shot type
        if best_closeup:
            add_5_star_rating_exiftool(best_closeup)
        if best_longshot:
            add_4_star_rating_exiftool(best_longshot)
        
        # Fallback logic if we don't have both a close-up and a long shot
        keepers = [k for k in [best_closeup, best_longshot] if k is not None]
        if len(keepers) < 2 and len(qualified_images) >= 2:
            qualified_images.sort(key=lambda x: x[1], reverse=True)
            top_two = [qualified_images[0][0], qualified_images[1][0]]
            
            # Rate the top two, ensuring we don't double-rate any already selected keepers
            for keeper_path in top_two:
                if keeper_path not in keepers:
                    # Determine if the fallback keeper is a close-up or long shot to assign the correct rating
                    is_closeup_fallback = any(img[2] for img in qualified_images if img[0] == keeper_path)
                    if is_closeup_fallback:
                        add_5_star_rating_exiftool(keeper_path)
                    else:
                        add_4_star_rating_exiftool(keeper_path)

    def process_images_thread(self):
        all_files = sorted([os.path.join(self.source_directory, f) for f in os.listdir(self.source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_files = len(all_files)
        start_time = time.time()

        current_set_data = []
        for i, image_path in enumerate(all_files):
            self.status_label.configure(text=f"Processing {i + 1}/{total_files}: {os.path.basename(image_path)}")
            
            if is_blank(image_path, self.blank_size_threshold):
                if current_set_data:
                    self.process_completed_set(current_set_data)
                current_set_data = []
            else:
                facial_features = analyze_facial_features(image_path)
                quality_metrics = analyze_image_quality(image_path)

                if facial_features and quality_metrics:
                    # Standard face-detected processing
                    eyes_score = facial_features['eyes_open'] * 20
                    sharpness_score = min(quality_metrics['sharpness'] / 133.0, 10)
                    contrast_score = min(quality_metrics['contrast'] / 10.0, 10)
                    brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
                    
                    # --- Quality Gate ---
                    if brightness_score < 4.0:
                        total_score = 0
                    else:
                        quality_score = sharpness_score + contrast_score + brightness_score
                        total_score = facial_features['smile_score'] + facial_features['head_turn_score'] + eyes_score + quality_score + facial_features.get('teeth_bonus', 0)
                    
                    write_grade_to_exif(image_path, total_score)
                    is_closeup = facial_features['face_size'] > self.close_up_threshold
                    current_set_data.append((image_path, total_score, is_closeup, facial_features['eyes_open']))
                
                elif not facial_features and quality_metrics and detect_person(image_path):
                    # Fallback 1: person detected but no face features
                    sharpness_score = min(quality_metrics['sharpness'] / 133.0, 10)
                    contrast_score = min(quality_metrics['contrast'] / 10.0, 10)
                    brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
                    
                    if brightness_score < 4.0:
                        total_score = 0
                    else:
                        quality_score = sharpness_score + contrast_score + brightness_score
                        base_score = 20  # Base score for person detection without face
                        total_score = base_score + quality_score
                    
                    write_grade_to_exif(image_path, total_score)
                    is_closeup = False  # Classify as long shot since no face detected
                    current_set_data.append((image_path, total_score, is_closeup, True))  # Assume eyes open since we can't detect
                
                elif not facial_features and not detect_person(image_path) and quality_metrics:
                    # Fallback 2: no face, no person detected, but substantial file size
                    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                    if file_size_mb >= self.blank_size_threshold:
                        sharpness_score = min(quality_metrics['sharpness'] / 133.0, 10)
                        contrast_score = min(quality_metrics['contrast'] / 10.0, 10)
                        brightness_score = max(0, 10 - abs(quality_metrics['brightness'] - 128) / 12.8)
                        
                        if brightness_score < 4.0:
                            total_score = 0
                        else:
                            quality_score = sharpness_score + contrast_score + brightness_score
                            base_score = 15  # Lower base score since no person detected
                            total_score = base_score + quality_score
                        
                        write_grade_to_exif(image_path, total_score)
                        is_closeup = False  # Classify as long shot 
                        current_set_data.append((image_path, total_score, is_closeup, True))  # Assume eyes open since we can't detect

            progress = (i + 1) / total_files
            self.progress_bar.set(progress)
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / (i + 1) if (i + 1) > 0 else 0
            remaining_files = total_files - (i + 1)
            time_remaining = time_per_file * remaining_files
            self.time_label.configure(text=f"Time Remaining: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")

        if current_set_data:
            self.process_completed_set(current_set_data)



        self.progress_bar.set(1.0)
        self.status_label.configure(text=f"Processing Complete. 5-star keepers rated in source directory.")
        self.process_button.configure(state="disabled")  # Disable after processing
        self.time_label.configure(text="")
        self.processing_complete = True
        self.set_ui_state("normal")

    def transfer_ratings_thread(self):
        self.status_label.configure(text="Starting rating transfer...")
        
        keeper_files = get_rated_keepers(self.source_directory)
        all_target_files = os.listdir(self.target_directory)
        
        if not keeper_files:
            self.status_label.configure(text="No 5-star rated keepers found in the source directory.")
            self.set_ui_state("normal")
            return
            
        transferred_count = 0
        for i, keeper_path in enumerate(keeper_files):
            keeper_basename = os.path.splitext(os.path.basename(keeper_path))[0]
            self.status_label.configure(text=f"Searching for {keeper_basename} in target directory...")
            
            matching_targets = [f for f in all_target_files if os.path.splitext(f)[0] == keeper_basename]
            
            if matching_targets:
                for target_filename in matching_targets:
                    target_path = os.path.join(self.target_directory, target_filename)
                    self.status_label.configure(text=f"Found. Applying 5-star rating to {target_filename}...")
                    add_5_star_rating_exiftool(target_path)
                    transferred_count += 1
            
            self.progress_bar.set((i + 1) / len(keeper_files))

        self.status_label.configure(text=f"Transfer Complete. {transferred_count} ratings applied.")
        self.set_ui_state("normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()
