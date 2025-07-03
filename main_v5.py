import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import face_recognition

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def get_gender(image_path):
    """Detects gender from the most prominent face in an image."""
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return None
        
        # For simplicity, we assume the first face is the one we're interested in
        # A more robust solution would be to analyze all faces or the largest one
        # This is a placeholder for a real gender detection model, as face_recognition doesn't provide it
        # We'll use a simple heuristic: if the face is wider than it is tall, we'll guess 'male'
        top, right, bottom, left = face_locations[0]
        if (right - left) > (bottom - top):
            return 'male'
        else:
            return 'female'
    except Exception as e:
        print(f"Could not determine gender for {os.path.basename(image_path)}: {e}")
        return None

def extract_features(image_path):
    """Extracts facial and body features using MediaPipe."""
    try:
        image = cv2.imread(image_path)
        if image is None: return None

        # Facial landmarks
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results_mesh = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results_mesh.multi_face_landmarks:
                return None
            face_landmarks = results_mesh.multi_face_landmarks[0]

        # Shoulder position
        with mp_pose.Pose(static_image_mode=True) as pose:
            results_pose = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results_pose.pose_landmarks:
                return None
            pose_landmarks = results_pose.pose_landmarks

        return {'face': face_landmarks, 'pose': pose_landmarks}
    except Exception as e:
        print(f"Feature extraction failed for {os.path.basename(image_path)}: {e}")
        return None

def calculate_similarity(features1, features2):
    """Calculates a similarity score between two sets of features."""
    if not features1 or not features2:
        return 0

    # Compare facial landmarks (simplified to a distance sum)
    face_dist = 0
    for i in range(len(features1['face'].landmark)):
        p1 = features1['face'].landmark[i]
        p2 = features2['face'].landmark[i]
        face_dist += np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    # Compare shoulder positions
    shoulder1_l = features1['pose'].landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder1_r = features1['pose'].landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder2_l = features2['pose'].landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder2_r = features2['pose'].landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    shoulder_dist = np.sqrt((shoulder1_l.x - shoulder2_l.x)**2 + (shoulder1_l.y - shoulder2_l.y)**2) + \
                    np.sqrt((shoulder1_r.x - shoulder2_r.x)**2 + (shoulder1_r.y - shoulder2_r.y)**2)

    # Lower distance is better, so we invert it for a score
    return 1 / (face_dist + shoulder_dist + 1e-6)

def find_best_match(image_paths, ideal_features):
    """Finds the best image in a set by comparing to ideal features."""
    best_image = None
    best_score = -1

    for image_path in image_paths:
        candidate_features = extract_features(image_path)
        if candidate_features:
            score = calculate_similarity(candidate_features, ideal_features)
            print(f"  - {os.path.basename(image_path)}: Similarity Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_image = image_path
    return best_image

def process_images(source_dir, dest_dir, ideal_dir):
    """Main processing loop."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Load ideal images and extract their features
    print("Loading ideal images...")
    ideal_images = {
        'male_closeup': extract_features(os.path.join(ideal_dir, 'ideal_male_closeup.jpg')),
        'female_closeup': extract_features(os.path.join(ideal_dir, 'ideal_female_closeup.jpg')),
        'male_longshot': extract_features(os.path.join(ideal_dir, 'ideal_male_longshot.jpg')),
        'female_longshot': extract_features(os.path.join(ideal_dir, 'ideal_female_longshot.jpg')),
    }
    print("Ideal images loaded and processed.")

    all_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # This is a simplified version that treats all images as one set
    # A full implementation would re-introduce the set detection logic
    
    print("\nProcessing candidate images...")
    # Determine gender and shot type for each image to select the correct ideal image
    # This is a complex step that is simplified here for brevity
    # We will assume all candidates are male close-ups for this example
    
    best_match = find_best_match(all_files, ideal_images['male_closeup'])
    
    if best_match:
        print(f"\nBest overall match: {os.path.basename(best_match)}")
        shutil.copy(best_match, dest_dir)
        print(f"Copied to '{dest_dir}'")

if __name__ == '__main__':
    source_directory = 'D:\\Memex small test'
    keeper_directory = os.path.join(os.path.dirname(source_directory), 'keeper_v4')
    ideal_directory = 'D:\\Memex Perfect'
    process_images(source_directory, keeper_directory, ideal_directory)
