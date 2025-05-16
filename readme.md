# Installation Instructions

## Step 1: Clone the Repository
```bash
git clone git@github.com:Summengardin/labshowcase.git
cd labshowcase
```

## Step 2: Set Up a Virtual Environment
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Step 4: Run the Script
To use the `basic_yolo_showcase.py` script, execute the following command:
```bash
python basic_yolo_showcase.py
```

This script demonstrates basic YOLO functionality for object detection, segmentation and pose estimation. Default configuration is to use webcam, but by specifying a `--source` this can be run on video file, image file, rtsp etc..  
```bash
python basic_yolo_showcase.py --source rtsp://10.1.3.72:8554/stream
```