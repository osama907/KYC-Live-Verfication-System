# Live Identity Verification Web App (Basic KYC System)

## Overview

This project is a **web-based identity verification system** designed to perform a basic **Know Your Customer (KYC)** verification using a user's webcam.

The application verifies a user's identity through the following steps:

1. **Face Detection** – Detects the user's face using the webcam.
2. **Liveness Check** – Ensures the user is a real person by asking them to perform an action (e.g., blink or turn their head).
3. **ID Card Capture** – The user holds their ID card in front of the camera.
4. **Face Matching** – The system compares the face from the webcam with the face printed on the ID card.
5. **Text Extraction** – The system extracts textual information from the ID card using OCR.

This system uses **Python for the backend** and **HTML/JavaScript for the frontend**.

---

# Features

* Real-time webcam access
* Face detection
* Liveness detection (blink / head movement)
* ID card capture
* Face recognition matching
* OCR text extraction from ID cards
* Simple web interface

---

# Technology Stack

## Backend

* Python
* Flask (API server)
* OpenCV (camera processing)
* face_recognition (facial matching)
* Mediapipe or Dlib (face landmarks / blink detection)
* Tesseract OCR (text extraction)

## Frontend

* HTML
* JavaScript
* WebRTC (webcam access)
* Fetch API for backend communication

---

# Project Structure

```
kyc-verification-app/
│
├── backend/
│   ├── app.py
│   ├── face_verification.py
│   ├── liveness_detection.py
│   ├── ocr_reader.py
│   └── utils.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── uploads/
│   ├── faces/
│   └── ids/
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/kyc-verification-app.git
cd kyc-verification-app
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
flask
opencv-python
face-recognition
mediapipe
pytesseract
numpy
Pillow
```

---

## 4. Install Tesseract OCR

Download Tesseract:

https://github.com/tesseract-ocr/tesseract

After installation, configure path if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

---

# Running the Application

Start the backend server:

```bash
python backend/app.py
```

Open the frontend:

```
frontend/index.html
```

Or host with Flask static serving.

Then open your browser:

```
http://localhost:5000
```

Allow **camera permissions** when prompted.

---

# Application Workflow

## Step 1 — Webcam Access

The browser requests webcam access using WebRTC.

## Step 2 — Face Detection

The system detects a face in the frame using OpenCV or Mediapipe.

## Step 3 — Liveness Detection

The user is prompted to perform actions such as:

* Blink
* Turn head left/right
* Smile

These actions confirm the user is a live person.

## Step 4 — ID Card Capture

The user holds their ID card in front of the webcam.

The system captures the frame and extracts:

* Face from ID
* Text information

## Step 5 — Face Matching

The system compares:

* Live webcam face
* ID card face

Using facial embeddings.

If similarity exceeds a threshold → **verification successful**.

---

# API Endpoints

### `/verify-liveness`

POST

Checks blink or head movement.

### `/upload-id`

POST

Uploads ID card image.

### `/compare-faces`

POST

Matches ID card face with live face.

### `/extract-text`

POST

Runs OCR on the ID card.

---

# Example Workflow Diagram

```
User Opens Page
        ↓
Webcam Activated
        ↓
Face Detection
        ↓
Liveness Check
        ↓
Capture ID Card
        ↓
Extract ID Face
        ↓
Face Comparison
        ↓
Extract ID Text
        ↓
Verification Result
```

---

# Security Considerations

* Use **HTTPS** in production.
* Store images securely.
* Delete temporary images after verification.
* Apply rate limiting.
* Use encrypted storage for sensitive data.

---

# Future Improvements

* Anti-spoofing with depth detection
* Passive liveness detection
* Document type detection
* NFC chip reading (for modern IDs)
* Mobile optimization
* AI fraud detection

---

# License

MIT License

---

# Disclaimer

This project is intended as a **basic demonstration of identity verification concepts** and should not be used as a complete production KYC solution without additional security, compliance, and regulatory checks.
