import os
import shutil
import sqlite3
import uvicorn
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace

# ---------------- APP INIT ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

DB_NAME = "database.db"
THRESHOLD = 0.75   # ✅ Tuned for CNIC verification

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

# ---------------- CNIC VALIDATION ----------------
def is_valid_cnic(image_path: str):
    print(f"\n--- 🔍 CHECKING CNIC: {os.path.basename(image_path)} ---")

    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Invalid image"

        h, w, _ = img.shape
        total_area = h * w

        # Aspect ratio (CNIC should be horizontal)
        if h > w * 1.2:
            print("❌ Vertical image detected")
            return False, "CNIC must be horizontal"

        # Detect face
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )

        if not faces:
            return False, "No face detected"

        face = faces[0]["facial_area"]
        face_area = face["w"] * face["h"]
        face_ratio = face_area / total_area

        print(f"👉 Face ratio: {face_ratio:.2f}")

        # Prevent selfies
        if face_ratio > 0.30:
            print("❌ Looks like selfie")
            return False, "Face too large (selfie detected)"

        print("✅ CNIC validation passed")
        return True, "Valid CNIC"

    except Exception as e:
        print(f"⚠️ CNIC validation error: {e}")
        return False, "Validation failed"

# ---------------- CNIC CHECK ENDPOINT ----------------
@app.post("/check_cnic")
def check_cnic(cnic_image: UploadFile = File(...)):
    temp_path = f"temp_{cnic_image.filename}"

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(cnic_image.file, f)

        valid, message = is_valid_cnic(temp_path)

        os.remove(temp_path)

        if valid:
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------- VERIFY IDENTITY ----------------
@app.post("/verify")
def verify_identity(
    cnic_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    try:
        cnic_path = os.path.join(UPLOAD_DIR, f"cnic_{cnic_image.filename}")
        live_path = os.path.join(UPLOAD_DIR, f"live_{live_image.filename}")

        with open(cnic_path, "wb") as f:
            shutil.copyfileobj(cnic_image.file, f)

        with open(live_path, "wb") as f:
            shutil.copyfileobj(live_image.file, f)

        print("\n--- 🤖 STARTING AI VERIFICATION ---")

        result = DeepFace.verify(
            img1_path=cnic_path,
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
        print("-----------------------------------")

        if verified:
            return {
                "status": "success",
                "message": "Identity Verified",
                "distance": distance
            }

        # Save for admin review
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            "INSERT INTO requests (cnic_path, live_path, status, distance) VALUES (?, ?, ?, ?)",
            (cnic_path, live_path, "PENDING", distance)
        )
        conn.commit()
        conn.close()

        return {
            "status": "pending",
            "message": "Verification sent for admin review",
            "distance": distance
        }

    except Exception as e:
        print(f"🔥 ERROR: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- ADMIN ROUTES ----------------
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
def admin_action(
    id: int = Form(...),
    action: str = Form(...)
):
    new_status = "APPROVED" if action == "approve" else "REJECTED"

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE requests SET status=? WHERE id=?", (new_status, id))
    conn.commit()
    conn.close()

    return {"status": "updated", "action": new_status}

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
