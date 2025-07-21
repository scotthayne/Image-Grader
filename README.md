# Image Grader and Selector

An automated tool for grading and selecting the best photos from image sets. The application analyzes facial features, image quality, and content to automatically identify and rate keeper images from photo sessions.

## Overview

The Image Grader processes folders of images, automatically:
- Detects photo sets using blank separator images
- Grades each image based on multiple criteria
- Selects the best close-up and long shot from each set
- Applies metadata ratings to keeper images
- Transfers ratings to corresponding RAW files

## Key Features

### Streamlined Workflow
- **No More Copying**: To significantly speed up processing, the application no longer copies keeper images to a separate folder.
- **Direct Rating**: 5-star ratings are applied directly to the original keeper images in the source directory.
- **Smart Transfer**: The rating transfer process now intelligently finds the 5-star rated images in the source directory to transfer their ratings.

### Intelligent Set Detection
- **Adjustable Blank Threshold**: GUI slider to set the file size threshold for blanks (3.0-7.0 MB, default 5.0 MB), with descriptive text to guide the user.
- **File Size + Content Analysis**: Identifies blank separator images using the adjustable file size and absence of people
- **Automatic Set Separation**: Organizes images into logical sets for comparison

### Advanced Image Grading (0-105 Points)
- **Facial Analysis (0-70 points)**:
  - Smile intensity: 0-50 points (using Mouth Aspect Ratio)
  - Eyes open: 20 points (binary, using Eye Aspect Ratio threshold of 0.14)
  - **Teeth bonus: 5 points** (NEW - awarded when teeth are visible)
- **Image Quality (0-30 points)**: Sharpness, contrast, and brightness analysis
- **Head positioning**: Additional scoring for proper head orientation

### Multi-Tier Image Classification
1. **Face Detected**: Full analysis with facial features and quality metrics
2. **Person Detected**: Fallback for distant shots where faces aren't detectable
3. **Content Detected**: File size-based fallback for substantial images (≥4.6 MB) without detectable people

### Differentiated Star Ratings
- **Close-ups:** Awarded a 5-star rating.
- **Long shots:** Awarded a 4-star rating.

### Smart Keeper Selection
- **Primary Logic**: Selects highest-scoring close-up and long shot from each set
- **Close-up Threshold**: Configurable face size percentage (default: 20% of image width)
- **Fallback Logic**: If primary logic fails, selects top 2 highest-scoring images regardless of type

### Quality Control
- **Universal Quality Gate**: A brightness threshold (default: 4.0) is applied to all images to prevent dark or poor-quality shots from being selected.

### Metadata Management
- **Grading**: Writes calculated scores to ImageDescription EXIF tag
- **Button State Management**: The "Grade and Select Images" button is now disabled after processing to prevent accidental re-runs, and re-enabled when a new source directory is selected.

- **Rating**: Applies 5-star ratings to keeper images
- **RAW Transfer**: Automatically finds and rates corresponding RAW files (e.g., .CR2) in target directory
- **Network Path Support**: Fixed compatibility with network drives and external storage

## Technical Implementation

### Core Technologies
- **GUI**: CustomTkinter for modern, user-friendly interface
- **Face Analysis**: MediaPipe for precise facial landmark detection
- **Image Processing**: OpenCV for quality analysis and person detection
- **Person Detection**: HOG (Histogram of Oriented Gradients) descriptor
- **Metadata**: ExifTool for robust EXIF data writing across all file formats

### Recent Improvements (Latest Session)
- **Enhanced Blank Detection**: Replaced brightness-based detection with file size + content analysis
- **Teeth Detection**: Added 5-point bonus for visible teeth using Mouth Aspect Ratio
- **Person Detection Fallback**: HOG-based detection for distant shots without visible faces  
- **Content Size Fallback**: Includes substantial images (with file size above the blank threshold) even without people detection
- **Network Path Fixes**: Resolved ExifTool compatibility issues with UNC paths using `os.path.abspath()`
- **Improved Long Shot Selection**: Multi-tier classification ensures both close-ups and long shots are available

## Usage

### Running the Application
```bash
# For development/local use
run_ImageGrader.bat

# For distribution
ImageGrader_Setup.exe
```

### Workflow
1. **Select Source Directory**: Choose folder containing image sets with blank separators
2. **Adjust Settings**: Configure close-up threshold (default: 20%)
3. **Process Images**: Click "Grade and Select Images" to analyze and select keepers
4. **Transfer Ratings**: (Optional) Select target directory to apply ratings to RAW files

### File Organization
- **Input**: Image sets separated by blank images in source directory
- **Output**: `keepers_rated/` folder containing selected images with 5-star ratings
- **Metadata**: All processed images receive numerical grades in ImageDescription field

## Requirements

- **Python 3.11+**
- **Virtual Environment**: Uses `.venv` for dependency management
- **ExifTool**: Required for metadata writing (included in installer)
- **Windows OS**: Current version optimized for Windows

### Key Dependencies
- `customtkinter` - Modern GUI framework
- `mediapipe` - Advanced facial analysis
- `opencv-python` - Image processing and person detection
- `Pillow` - Image quality analysis
- `numpy` - Numerical computations

## Installation

### For End Users
1. Download and run `ImageGrader_Setup.exe`
2. Application installs with all dependencies included

### For Development
1. Clone repository
2. Create virtual environment: `uv venv`
3. Install dependencies: `uv pip install -r requirements.txt`
4. Run: `run_ImageGrader.bat`

## Scoring System Details

### Face-Detected Images (0-105 points)
- Smile Analysis: 0-50 points
- Eyes Open: 0-20 points  
- Teeth Bonus: 0-5 points
- Image Quality: 0-30 points

### Person-Detected Images (0-50 points)
- Base Person Score: 20 points
- Image Quality: 0-30 points

### Content-Only Images (0-45 points)
- Base Content Score: 15 points
- Image Quality: 0-30 points

## Future Enhancements

- Additional facial expression analysis
- Batch processing optimization
- Cloud storage integration
- Custom scoring weight configuration
- Multiple person detection and selection

## Development History

Built with iterative development using AI assistance, focusing on:
- Robust error handling and edge case management
- User experience optimization
- Cross-platform compatibility considerations
- Comprehensive testing and validation

---

*Generated with Memex AI Assistant*