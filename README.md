# Real-Time ANPR System with YOLOv8, EasyOCR, and Multithreading

A real-time Automatic Number Plate Recognition (ANPR) system using [YOLOv8](https://github.com/ultralytics/ultralytics) for license plate detection and [EasyOCR](https://github.com/JaidedAI/EasyOCR) for character recognition. The system features a multithreaded CPU pipeline for efficient video processing, supports GPU acceleration for inference, and includes custom logic for multiple plate types (such as Kuwaiti plates).

![Screenshot 2025-06-20 224740](https://github.com/user-attachments/assets/86d604db-2225-4130-8980-3c290634d049)


## Features

- **Real-time** processing from video streams or files
- **YOLOv8** for robust, fast license plate detection (GPU-supported)
- **EasyOCR** for accurate plate number reading
- **Multithreaded** CPU pipeline for efficient, parallel video, detection, and OCR handling
- **Custom plate format support** for region-specific license plates (e.g., Kuwaiti plates)
- Clean, modular Python code

## Example Workflow

1. **Detection Thread:** Captures and detects plates from video frames using YOLOv8.
2. **OCR Thread:** Crops plates and recognizes text using EasyOCR.
3. **Display Thread:** Shows results in real-time.

## Directory Structure

├── main.py

├── queues_functions.py

├── utils.py

├── saved_images/ # (auto-generated, not tracked) - stores recognized plates

├── .gitignore

└── README.md



## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- OpenCV (`opencv-python`)
- NumPy

### Installation

git clone https://github.com/Okh2891996/anpr-yolov8-easyocr-multithread.git
cd anpr-yolov8-easyocr-multithread

# Install dependencies
pip install -r requirements.txt


# Usage
1 . Place your trained YOLOv8 license plate detection model (e.g., ANPR.pt) in the project directory.

2 . Place your video file (e.g., license_plate.mp4) in the project directory.

3. Run the main script:

python main.py


4. A display window will show real-time results. Press Q to quit.

Note:

Detected plate images and results are saved in the saved_images/ directory.


The system checks for GPU at startup and uses it if available.

# Applications
Parking and access control automation

Traffic monitoring and law enforcement

Security and surveillance systems

# Note 
Contributions, bug reports, and feedback are welcome!
