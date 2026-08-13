# 🚗 Vietnamese License Plate Recognition — OpenCV + KNN

A classical image processing pipeline (no deep learning) to detect, extract, and read Vietnamese vehicle license plates, supporting both **motorbike** and **car** plates.

## 📌 Overview

The system performs 3 main stages:

1. **Plate detection & extraction** — locate and crop the license plate region from the input image
2. **Character segmentation** — isolate individual characters on the plate
3. **Character recognition** — classify each segmented character using a **K-Nearest Neighbors (KNN)** model trained on a pre-built character dataset

Two separate pipelines are implemented, tuned to each vehicle type's plate geometry:

| Notebook | Vehicle | Plate type |
|---|---|---|
| `test.ipynb` | Motorbike | Square/2-row plates (HSV Top-Hat/Black-Hat → adaptive threshold → Canny) |
| `test1.ipynb` | Car | Rectangular/1-row plates (Sobel edge detection → Otsu threshold → morphological closing) |

Built entirely with classical CV techniques, making it lightweight, easy to understand, and runnable on low-spec hardware — ideal for students learning the fundamentals of computer vision.

## 🔧 Pipeline

### Stage 1 — Plate Detection & Extraction

| Step | Motorbike (`test.ipynb`) | Car (`test1.ipynb`) |
|---|---|---|
| Resize | 1920×1080 | 600×400 |
| Grayscale | HSV → V channel + Top-Hat − Black-Hat | RGB → Gray + histogram equalization |
| Edge/contrast enhancement | Morphological Top-Hat / Black-Hat | Sobel operator (x-direction) |
| Denoise + Binarize | Gaussian Blur → Adaptive Threshold (Gaussian) | Otsu's threshold |
| Edge/shape refinement | Canny edge detection → dilation | Morphological closing (17×3 rectangular kernel) |
| Candidate region search | `findContours` → filter to 4-sided polygons matching plate width/height/aspect ratio | `findContours` → filter by min/max width and aspect ratio ≥ 3.1 (rectangular car plates) |

### Stage 2 — Character Segmentation

For each candidate plate ROI (resized to 600×200):

1. Re-binarize (histogram equalization + threshold)
2. Find character contours (`RETR_TREE`)
3. Filter candidates by:
   - **Area** (not too small/large relative to plate size)
   - **Height/width bounds** (characters are taller than wide)
   - **Aspect ratio**
   - **White-pixel density** (rejects near-solid or near-empty blobs)
4. Sort remaining contours left-to-right (and top-to-bottom for 2-row plates) to preserve reading order

### Stage 3 — Character Recognition (KNN)

- Each segmented character is resized to **20×30** and flattened into a feature vector
- Classified using **OpenCV's `cv.ml.KNearest`** (k=5), trained on `flattened_images.txt` / `classifications.txt`
- Predicted characters are concatenated into the final plate string

## 🛠️ Tech Stack

- **Python**
- **OpenCV** (`cv2`) — image processing, morphological transforms, contour detection, KNN classifier
- **NumPy**, **Matplotlib**
- Classical computer vision + KNN only — no CNN/deep learning model involved

## ✅ Strengths

- Simple to implement and deploy; good for learning core image processing/CV concepts
- Lightweight — runs smoothly even on low-spec hardware, unlike CNN/SVM-based approaches
- Handles both motorbike and car plate geometries with tailored pipelines
- Well suited for students building foundational understanding of image processing and AI

## ⚠️ Limitations

- Sensitive to environmental conditions: reflections, motion blur, glare, unclear characters → misdetection
- Requires a fairly controlled setup (fixed camera, controlled lighting/background) for stable results
- Detection thresholds (width/height/aspect ratio ranges) are hand-tuned per vehicle type and may not generalize to all plate variants
- Not fully automatic — still requires some human oversight

## 🚀 Future Work

- Replace KNN character recognition with a stronger model (SVM/CNN) or a modern detector (YOLO/YOLOv3) for higher accuracy and generalization
- Use specialized cameras robust to fog, night conditions, and glare
- Improve plate localization using Hough Transform, color-based detection, and motion-blur reduction techniques
- Integrate into real-world systems: parking/warehouse management, vehicle tracking, lost-vehicle search

## 🎓 Academic Context

Report: *"Nhận dạng và đọc biển số xe"* (License Plate Detection and Reading)
Course: **Phân tích xử lý ảnh** (Image Processing Analysis)
Faculty of Mathematics & Computer Science, VNUHCM – University of Science, Class 23TTH1

**Team:** Đoàn Anh Quân (23110111), Trần Tấn Hiệp (23110082)
**Instructor:** Thầy Huỳnh Thanh Sơn
