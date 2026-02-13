import os
import shutil
import sqlite3
import cv2
import re
import pytesseract
import numpy as np
import uvicorn
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace

# ---------------- APP INIT ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_NAME = "database.db"

# --- SMART THRESHOLD ---
AUTO_PASS_THRESHOLD = 0.42      
ADMIN_REVIEW_THRESHOLD = 0.60   
MODELS = ["ArcFace", "Facenet512"]

# ---------------- DB INIT & MIGRATION ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table with all required columns
    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cnic_path TEXT,
            selfie_path TEXT,
            status TEXT,
            distance REAL,
            doc_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check for missing columns (Migration Helper)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("PRAGMA table_info(requests)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'created_at' not in columns:
        conn.execute("ALTER TABLE requests ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if 'selfie_path' not in columns:
        conn.execute("ALTER TABLE requests ADD COLUMN selfie_path TEXT")
        
    conn.commit()
    conn.close()

init_db()

# ---------------- HELPERS ----------------
def extract_expiry_date(text):
    date_patterns = [r"(\d{2}[./-]\d{2}[./-]\d{4})", r"(\d{4}[./-]\d{2}[./-]\d{2})"]
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            clean = re.sub(r"[/-]", ".", m)
            found_dates.append(clean)
    return found_dates[-1] if found_dates else None

def extract_text_robust(image_path):
    img = cv2.imread(image_path)
    if img is None: return ""
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh1, config='--oem 3 --psm 11')
    if not extract_expiry_date(text):
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text += " " + pytesseract.image_to_string(thresh2, config='--oem 3 --psm 11')
    return re.sub(r"\s+", " ", text).strip()

def is_duplicate_document(new_doc_path):
    """Checks if the face on the ID already exists in an APPROVED record."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT cnic_path FROM requests WHERE status='APPROVED'")
    rows = c.fetchall()
    conn.close()

    for row in rows:
        existing_path = row['cnic_path']
        if not os.path.exists(existing_path): continue
        try:
            # Using ArcFace for quick ID-to-ID comparison
            res = DeepFace.verify(
                img1_path=new_doc_path, 
                img2_path=existing_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )
            if res["distance"] <= 0.35: # Threshold for identical ID detection
                return True
        except: continue
    return False

# ---------------- VERIFY ENDPOINT ----------------
@app.post("/verify")
async def verify_identity(
    cnic_image: UploadFile = File(...), 
    selfie_image: UploadFile = File(...) 
):
    try:
        print("\n" + "🚀" * 15)
        print("AI SYSTEM: NEW VERIFICATION REQUEST")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_path = os.path.join(UPLOAD_DIR, f"doc_{timestamp}.jpg")
        selfie_path = os.path.join(UPLOAD_DIR, f"live_{timestamp}.jpg")

        for file, path in [(cnic_image, doc_path), (selfie_image, selfie_path)]:
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        # --- STEP 0: DUPLICATE CHECK ---
        print("🔍 Checking for duplicate ID...")
        if is_duplicate_document(doc_path):
            print("❌ REJECTED: Duplicate Document Detected")
            return {"status": "REJECTED", "message": "This document is already registered."}

        # 1. OCR Stage
        raw_text = extract_text_robust(doc_path)
        expiry_str = extract_expiry_date(raw_text)
        doc_type = "Passport" if "PASSPORT" in raw_text.upper() or "<<" in raw_text else "ID Card"
        
        # 2. Face Verification
        distances = []
        for model in MODELS:
            res = DeepFace.verify(
                img1_path=doc_path, img2_path=selfie_path, 
                model_name=model, detector_backend="retinaface",
                distance_metric="cosine", align=True
            )
            distances.append(float(res["distance"]))
        
        avg_distance = sum(distances) / len(distances)

        # 3. Decision Logic
        final_status = "REJECTED"
        icon = "❌"
        if avg_distance <= AUTO_PASS_THRESHOLD:
            final_status = "APPROVED"
            icon = "✅"
        elif avg_distance <= ADMIN_REVIEW_THRESHOLD:
            final_status = "PENDING"
            icon = "⚠️"

        print(f"📊 Distance: {avg_distance:.4f}")
        print(f"{icon} STATUS  : {final_status}")

        # 4. Save to DB
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""INSERT INTO requests (cnic_path, selfie_path, status, distance, doc_type) 
                     VALUES (?, ?, ?, ?, ?)""",
                  (doc_path, selfie_path, final_status, avg_distance, doc_type))
        conn.commit()
        last_id = c.lastrowid
        conn.close()

        print(f"🆔 DB ID    : {last_id}")
        print("🚀" * 15 + "\n")

        return {
            "id": last_id,
            "status": final_status,
            "distance": avg_distance,
            "doc_type": doc_type,
            "expiry": expiry_str
        }

    except Exception as e:
        print(f"🔥 ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

# ---------------- ADMIN SECTION ----------------
@app.get("/admin/all")
def admin_all():
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM requests ORDER BY created_at DESC").fetchall()
    return {"requests": [dict(row) for row in rows]}

@app.get("/admin/pending")
def admin_pending():
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM requests WHERE status='PENDING' ORDER BY created_at DESC").fetchall()
    return {"requests": [dict(row) for row in rows]}

@app.post("/admin/action")
def admin_action(id: int = Form(...), action: str = Form(...)):
    new_status = "APPROVED" if action == "approve" else "REJECTED"
    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE requests SET status=? WHERE id=?", (new_status, id))
    conn.commit()
    print(f"🔨 ADMIN: Request #{id} manually {new_status}")
    return {"status": "updated"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)