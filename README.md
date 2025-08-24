
# Facial Attendance Recognizer

A **facial recognition-based attendance system** built with **Python, OpenCV, and Tkinter**.
This application allows you to:

* Register new people by capturing their face images.
* Train a face recognition model.
* Recognize faces in real-time using a webcam.
* Log attendance with timestamps.

---

## ✨ Features

* **GUI-based** (Tkinter) for easy interaction.
* **Add Person** → capture 500+ face samples for training.
* **Train & Recognize** → trains a model and detects faces in real-time.
* **Attendance Logging** → saves names with timestamps to `attendance/attendances.txt`.
* **Face Detection** → uses Haar Cascade Classifiers.
* **Face Recognition** → based on OpenCV’s LBPH (Local Binary Pattern Histogram).

---

## 📦 Requirements

Make sure you have the following installed:

* Python 3.x
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Pillow](https://pypi.org/project/Pillow/)
* [Numpy](https://pypi.org/project/numpy/)

Install dependencies:

```bash
pip install opencv-python opencv-contrib-python pillow numpy
```

⚠️ Note: `opencv-contrib-python` is required for `cv2.face.LBPHFaceRecognizer_create()`.

---

## ▶️ Usage

### 1. Run the program

```bash
python facial_attendance.py
```

### 2. Add a Person

* Click **Add Person**.
* Enter a name.
* The system will capture face images from your webcam and store them under `images/<name>/`.

### 3. Train & Recognize

* Click **Recognize**.
* The system will:

  * Train the face recognizer using collected images.
  * Start the webcam for real-time recognition.
  * Display the recognized name on screen.
  * Log recognized names with timestamps into `attendance/attendances.txt`.

### 4. Stop Recognition

* Press **Q** to quit recognition.

---

## 📖 File Structure

```
project/
│── facial_attendance.py      # Main script
│── cascades/                 # Haar Cascade XML files
│── images/                   # Stored face images
│   └── <person_name>/
│── pickles/                  # Encoded labels
│── recognizers/              # Trained model (face-trainner.yml)
│── attendance/               # Attendance logs
│   └── attendances.txt
```

---

## ⚠️ Notes

* Ensure your **webcam is enabled**.
* Collect at least **500+ face images** per person for accurate recognition.
* Run `Recognize` after adding people so the model can retrain.
* Attendance is **overwritten** each session (app writes to `attendances.txt`). You may modify the script to **append** instead of overwrite.

---

## 🛠️ Future Improvements

* Append attendance instead of overwriting.
* Export attendance to **Excel/CSV**.
* Improve GUI design.
* Add support for multiple cameras.
* Optimize face detection for speed.

---

## 👨‍💻 Author

Made by **Abishek Ganesh**
Version: `0.01`

---

Would you like me to also add a **setup guide** (with folder structure creation and cascade downloads), so users can run it without errors on a fresh system?
