# 🚗 Real-Time Vehicle Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge&logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **A Computer Vision project capable of detecting and classifying vehicles (Cars, Buses, Trucks, Motorcycles, Ambulances) in real-time video feeds using the YOLOv8 architecture.**


---

## 📝 About the Project

Traffic management is a critical challenge in modern urban planning. This project leverages **Deep Learning** to automate the detection of vehicles from video footage. Unlike standard models, this system has been **custom-trained** to specifically distinguish between 5 distinct classes of vehicles, providing granular data for traffic analysis.

The system processes video input frame-by-frame, applies the YOLOv8 inference engine, and outputs a video file with bounding boxes and confidence scores drawn around detected vehicles.

---

## ✨ Key Features

* **🔍 Multi-Class Detection:** Identifies 5 specific classes: `Ambulance`, `Bus`, `Car`, `Motorcycle`, `Truck`.
* **⚡ Real-Time Performance:** Optimized for speed using the YOLOv8 Nano model.
* **🎥 Video Pipeline:** Seamlessly reads raw video, processes it, and saves the labeled output automatically.
* **📊 Confidence Scoring:** Displays probability scores (e.g., "Bus 0.92") to indicate detection reliability.
* **🛡️ Robustness:** trained with data augmentation (Mosaic, scaling) to handle occlusion and varying lighting conditions.

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **Model Architecture** | YOLOv8 (Ultralytics) |
| **Image Processing** | OpenCV (cv2) |
| **Training Environment** | Local CPU/GPU |
| **IDE** | VS Code |

---

## 📂 Project Structure

```bash
Vehicle-Detection-System/
│
├── datasets/              # (Ignored by Git) Contains raw images & labels
├── models/
│   └── best.pt            # 🧠 The custom trained YOLOv8 model
├── detect_vehicles.py     # 🚀 Main script to run detection on videos
├── train_model.py         # 🎓 Script to retrain the model
├── data.yaml              # ⚙️ Dataset configuration file
├── requirements.txt       # 📦 List of dependencies
├── README.md              # 📄 Project documentation
├── statement.md           # 📄 Problem statement & scope
└── .gitignore             # 🚫 Files to exclude (videos, datasets)
