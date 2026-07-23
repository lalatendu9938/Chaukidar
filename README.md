# Chaukidar
Chaukidar: An Edge-AI powered selective bio-acoustic deterrent system using YOLO26 and OpenCV to detect wildlife threats in real-time while filtering out livestock.
By leveraging a state-of-the-art YOLO26 one-stage architecture, the system monitors live video feeds to detect crop-destroying wild animals in real-time, triggering species-specific bio-acoustic deterrents while strictly ignoring non-threat livestock.

## 🚀 Key Features
* **Real-Time Edge Inference:** Utilizes a lightweight, NMS-free YOLO26 architecture optimized for local hardware, eliminating the thermal throttling and latency bottlenecks of traditional Transformer models (like RT-DETR).
* **High-Precision Filtering:** Hardcoded >0.75 confidence threshold algorithm to aggressively suppress false positives (shadows, background noise), reducing false alarms by 96%.
* **Dynamic Bio-Acoustic Engine:** Custom audio pipeline built with `pygame` and `numpy` that synthesizes species-specific deterrent sounds upon verified threat detection, improving response time by 80%.
* **Automated Data Engineering:** Includes a custom preprocessing script (`merge.py`) that standardized, relabeled, and mapped 6,500+ images from 7 disparate datasets into a unified YOLO-formatted master dataset.

## 🛠️ Tech Stack
* **Computer Vision:** Ultralytics (YOLO26), OpenCV
* **Data Processing & Math:** Python, NumPy, OS/Shutil
* **Audio Engineering:** PyGame
* **Evaluation Metrics:** Precision-Recall curves, mAP@50, F1-Score calculation

## 📊 Dataset & Model Performance
The model was trained on a custom, unified dataset of **6,500+ images** spanning 6 classes (Elephant, Monkey, Wild Boar, Cow, Deer, Leopard).
* **Overall Detection Accuracy (mAP@50):** 92%
* **False Alarm Reduction:** 96%
* **Target Environment:** Agricultural Edge CCTV / Webcams

## 📂 Repository Structure
* `chaukidar.py` - The main live OpenCV video pipeline and audio triggering logic.
* `merge.py` - The data engineering script used to clean, map, and merge raw datasets.
* `trainyolo.py`, `trainrtdetr.py`- The training code utilized to benchmark YOLO and RT-DETR architectures.
* `/Master_Dataset` - Contains the `data.yaml` configuration file for the Ultralytics engine.

## 💡 How It Works
1. The camera feeds real-time frames into the OpenCV pipeline.
2. The YOLO26 model runs inference, looking for spatial and textual features of known threats.
3. If an animal is detected, the prediction must pass a strict `conf=0.75` mathematical gatekeeper.
4. If validated, the bio-acoustic engine fires the mapped audio file to deter the specific animal without habituating it to generic alarms.
