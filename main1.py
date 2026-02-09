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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_NAME = "database.db"

# --- SMART THRESHOLD LOGIC ---
VERIFIED_THRESHOLD = 0.55  # Auto-pass if below this
DOUBT_THRESHOLD = 0.75     # Send to Admin if between 0.55 and 0.75. Reject if above.

MODELS = ["ArcFace", "Facenet512"]

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
            distance REAL,
            doc_type TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def fix_database():
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN doc_type TEXT")
        conn.commit()
        print("✅ Database updated successfully!")
    except sqlite3.OperationalError:
        print("ℹ️ Column already exists.")
    conn.close()

fix_database()

# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    enhanced = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
    cv2.imwrite(image_path, enhanced)
    return enhanced

# ---------------- SMART FACE SELECTOR ----------------
def get_primary_face(image_path, label="Image"):
    img = cv2.imread(image_path)
    if img is None: return None
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path, 
            detector_backend="retinaface", 
            enforce_detection=True,
            align=True
        )
        main_face = max(faces, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
        return main_face
    except:
        return None

# ---------------- OCR HELPER ----------------
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh)
    return re.sub(r"\s+", " ", text).strip()

# ---------------- VERIFY ENDPOINT ----------------
@app.post("/verify")
def verify_identity(cnic_image: UploadFile = File(...), live_image: UploadFile = File(...)):
    try:
        doc_path = os.path.join(UPLOAD_DIR, f"doc_{cnic_image.filename}")
        live_path = os.path.join(UPLOAD_DIR, f"live_{live_image.filename}")

        with open(doc_path, "wb") as f: shutil.copyfileobj(cnic_image.file, f)
        with open(live_path, "wb") as f: shutil.copyfileobj(live_image.file, f)

        enhance_image(doc_path)
        enhance_image(live_path)

        print("\n--- 🤖 AI PROCESSING ---", flush=True)

        # 1. Document Check
        text = extract_text(doc_path)
        face_doc = get_primary_face(doc_path, "Document")
        
        if not face_doc:
            return {"status": "error", "message": "Invalid Document - No Face Detected"}

        doc_type = "Passport" if "PASSPORT" in text.upper() or "<<" in text else "ID Card"

        # 2. Live Face Check
        face_live = get_primary_face(live_path, "Live Photo")
        if not face_live:
            return {"status": "error", "message": "Face unclear - Look at the camera"}

        # 3. Verification Calculation
        distances = []
        for model in MODELS:
            res = DeepFace.verify(
                img1_path=doc_path, 
                img2_path=live_path, 
                model_name=model, 
                detector_backend="retinaface",
                distance_metric="cosine",
                enforce_detection=True
            )
            distances.append(float(res["distance"]))

        avg_distance = sum(distances) / len(distances)
        print(f"📊 Avg Distance: {avg_distance:.4f}", flush=True)

        # --- REFINED DECISION LOGIC ---
        
        # CASE A: Strong Match
        if avg_distance <= VERIFIED_THRESHOLD:
            print("✅ Result: AUTO-PASSED", flush=True)
            return {
                "status": "success", 
                "verified": True, 
                "document": doc_type, 
                "distance": avg_distance,
                "message": "Identity Verified Successfully"
            }

        # CASE B: High Doubt (Send to Admin)
        elif VERIFIED_THRESHOLD < avg_distance <= DOUBT_THRESHOLD:
            print("💾 Result: IN DOUBT - Sent to Admin", flush=True)
            conn = sqlite3.connect(DB_NAME); c = conn.cursor()
            c.execute("INSERT INTO requests (cnic_path, live_path, status, distance, doc_type) VALUES (?, ?, ?, ?, ?)",
                      (doc_path, live_path, "PENDING", avg_distance, doc_type))
            conn.commit(); conn.close()
            return {
                "status": "pending", 
                "verified": False, 
                "document": doc_type, 
                "distance": avg_distance,
                "message": "Verification is under review by an administrator."
            }

        # CASE C: Complete Mismatch
        else:
            print("❌ Result: REJECTED (Mismatch)", flush=True)
            return {
                "status": "rejected", 
                "verified": False, 
                "document": doc_type, 
                "distance": avg_distance,
                "message": "Face mismatch. Identity could not be verified."
            }

    except Exception as e:
        print(f"🔥 FATAL ERROR: {str(e)}", flush=True)
        return {"status": "error", "message": "Processing Error"}

# ---------------- ADMIN ENDPOINTS ----------------
@app.get("/admin/pending")
def admin_pending():
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE status='PENDING'")
    rows = c.fetchall(); conn.close()
    return {"requests": [dict(row) for row in rows]}

@app.post("/admin/action")
def admin_action(id: int = Form(...), action: str = Form(...)):
    new_status = "APPROVED" if action == "approve" else "REJECTED"
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE requests SET status=? WHERE id=?", (new_status, id))
    conn.commit(); conn.close()
    return {"status": "updated", "final_decision": new_status}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)