import os
import shutil
import sqlite3
import cv2
import re
import pytesseract
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace

# ---------------- APP INIT ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

DB_NAME = "database.db"
THRESHOLD = 0.75

# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cnic_path TEXT,
            live_path TEXT,
            status TEXT,
            distance REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- OCR HELPER ----------------
def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    text = pytesseract.image_to_string(gray)
    return re.sub(r"\s+", " ", text).strip()

# ---------------- DOCUMENT VALIDATION ----------------
def validate_document(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return False, None

    h, w, _ = img.shape
    total_area = h * w

    text = extract_text(img)
    if len(text) < 15:
        return False, None

    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="retinaface",
        enforce_detection=True,
        align=True
    )

    if len(faces) != 1:
        return False, None

    face = faces[0]["facial_area"]
    face_ratio = (face["w"] * face["h"]) / total_area

    # Passport (MRZ detection)
    if "<<<<" in text or "<<<" in text:
        if face_ratio <= 0.35:
            return True, "Passport"
        return False, None

    # CNIC (horizontal)
    if h <= w * 1.2 and face_ratio <= 0.30:
        return True, "CNIC"

    return False, None

# ---------------- BASIC LIVENESS ----------------
def check_liveness(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    print(f"📱 Laplacian variance: {laplacian_var:.2f}")
    print(f"📐 Edge density: {edge_density:.4f}")

    # NEW: Less strict thresholds for real face cameras
    if laplacian_var < 15 and edge_density < 0.005:
        return False, laplacian_var, edge_density  # Spoof
    return True, laplacian_var, edge_density


# ---------------- VERIFY ENDPOINT ----------------
@app.post("/verify")
def verify_identity(
    cnic_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    try:
        doc_path = os.path.join(UPLOAD_DIR, f"doc_{cnic_image.filename}")
        live_path = os.path.join(UPLOAD_DIR, f"live_{live_image.filename}")

        with open(doc_path, "wb") as f:
            shutil.copyfileobj(cnic_image.file, f)

        with open(live_path, "wb") as f:
            shutil.copyfileobj(live_image.file, f)

        print("\n--- 🤖 STARTING AI VERIFICATION ---")

        # -------- STEP 1: DOCUMENT CHECK --------
        valid_doc, doc_type = validate_document(doc_path)
        if not valid_doc:
            print("❌ INVALID DOCUMENT")
            return {
                "status": "error",
                "message": "Please upload a valid CNIC or Passport"
            }

        print(f"✅ Document detected: {doc_type}")

        # -------- STEP 2: LIVENESS CHECK --------
        live_ok, lap, edge = check_liveness(live_path)

        if not live_ok:
            print("❌ SPOOF ATTACK DETECTED")
            return {
                "status": "error",
                "message": "Live face not detected. Please use a real person."
            }

        print("✅ LIVENESS PASSED")

        # -------- STEP 3: FACE VERIFICATION --------
        result = DeepFace.verify(
            img1_path=doc_path,
            img2_path=live_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )

        distance = float(result["distance"])
        verified = distance < THRESHOLD

        print(f"📊 Distance: {distance:.4f}")
        print(f"🏆 VERIFIED: {verified}")
        print("---------------------------------------")

        if verified:
            return {
                "status": "success",
                "verified": True,
                "document": doc_type,
                "distance": distance
            }

        # Save for admin review
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            "INSERT INTO requests (cnic_path, live_path, status, distance) VALUES (?, ?, ?, ?)",
            (doc_path, live_path, "PENDING", distance)
        )
        conn.commit()
        conn.close()

        return {
            "status": "pending",
            "verified": False,
            "document": doc_type,
            "distance": distance
        }

    except Exception as e:
        print(f"🔥 ERROR: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- ADMIN ----------------
@app.get("/admin/pending")
def admin_pending():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE status='PENDING'")
    rows = c.fetchall()
    conn.close()
    return {"requests": [dict(row) for row in rows]}

@app.post("/admin/action")
def admin_action(id: int = Form(...), action: str = Form(...)):
    new_status = "APPROVED" if action == "approve" else "REJECTED"
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE requests SET status=? WHERE id=?", (new_status, id))
    conn.commit()
    conn.close()
    return {"status": "updated", "action": new_status}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
