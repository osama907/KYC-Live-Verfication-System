import os
import shutil
import cv2
import re
import pytesseract
import numpy as np
import uvicorn
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Load environment variables (MONGODB_URL)
load_dotenv()

# --- CONFIGURATION ---
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB Limit to protect Hostinger VPS memory
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Osama Ali Shah KYC System")

# CORS Setup for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder so admin can view images
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- MONGODB SETUP ---
MONGO_URL = os.getenv("MONGODB_URL")
client = AsyncIOMotorClient(MONGO_URL)
db = client.get_database("kyc_database")
requests_collection = db.get_collection("requests")

# AI Settings
AUTO_PASS_THRESHOLD = 0.42
ADMIN_REVIEW_THRESHOLD = 0.60
MODELS = ["ArcFace", "Facenet512"]

# --- SECURITY & UTILS ---

async def validate_file(file: UploadFile):
    """Checks file type and size before processing."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Only image files are allowed.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, detail="File too large (Max 5MB).")
    await file.seek(0)

def extract_text_robust(image_path):
    """OCR to extract data from ID card."""
    img = cv2.imread(image_path)
    if img is None: return ""
    # Image enhancement for better OCR
    gray = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2), cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0).apply(gray)
    text = pytesseract.image_to_string(enhanced, config='--oem 3 --psm 11')
    return re.sub(r"\s+", " ", text).strip()

async def is_duplicate_document(new_doc_path):
    """Check if this face already exists in APPROVED records."""
    cursor = requests_collection.find({"status": "APPROVED"})
    async for row in cursor:
        try:
            res = DeepFace.verify(img1_path=new_doc_path, img2_path=row['cnic_path'], 
                                  model_name="ArcFace", enforce_detection=False)
            if res["distance"] <= 0.35: return True
        except: continue
    return False

# --- THE HEAVY AI WORKER (Background) ---

async def run_ai_verification_task(req_id: str, doc_path: str, selfie_path: str):
    """Processes AI matching without blocking the main server."""
    try:
        logging.info(f"Starting AI Task for ID: {req_id}")
        
        # 1. Duplicate Check
        if await is_duplicate_document(doc_path):
            await requests_collection.update_one({"_id": ObjectId(req_id)}, 
                {"$set": {"status": "REJECTED", "message": "Duplicate ID detected."}})
            return

        # 2. Face Verification (Multi-Model)
        distances = []
        for model in MODELS:
            res = DeepFace.verify(img1_path=doc_path, img2_path=selfie_path, 
                                  model_name=model, detector_backend="retinaface", enforce_detection=False)
            distances.append(float(res["distance"]))
        
        avg_dist = sum(distances) / len(distances)

        # 3. Final Decision Logic
        final_status = "REJECTED"
        if avg_dist <= AUTO_PASS_THRESHOLD:
            final_status = "APPROVED"
        elif avg_dist <= ADMIN_REVIEW_THRESHOLD:
            final_status = "PENDING"

        # Update MongoDB with results
        await requests_collection.update_one(
            {"_id": ObjectId(req_id)}, 
            {"$set": {"status": final_status, "distance": avg_dist, "processed_at": datetime.utcnow()}}
        )
        logging.info(f"Task {req_id} finished with status: {final_status}")

    except Exception as e:
        logging.error(f"AI Task Error for {req_id}: {str(e)}")
        await requests_collection.update_one({"_id": ObjectId(req_id)}, 
            {"$set": {"status": "ERROR", "message": "Internal AI Processing Error"}})

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Checks if server and DB are online."""
    try:
        await client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except:
        return {"status": "unhealthy", "database": "disconnected"}

@app.post("/verify")
async def verify_identity(bg: BackgroundTasks, cnic: UploadFile = File(...), selfie: UploadFile = File(...)):
    """User uploads ID and Selfie. Returns ID immediately."""
    await validate_file(cnic)
    await validate_file(selfie)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_path = os.path.join(UPLOAD_DIR, f"d_{timestamp}.jpg")
    selfie_path = os.path.join(UPLOAD_DIR, f"s_{timestamp}.jpg")

    # Save images
    with open(doc_path, "wb") as f: shutil.copyfileobj(cnic.file, f)
    with open(selfie_path, "wb") as f: shutil.copyfileobj(selfie_image.file, f)

    # Insert initial record
    new_request = {
        "status": "PROCESSING",
        "cnic_path": doc_path,
        "selfie_path": selfie_path,
        "created_at": datetime.utcnow()
    }
    result = await requests_collection.insert_one(new_request)
    request_id = str(result.inserted_id)

    # Queue the heavy AI work
    bg.add_task(run_ai_verification_task, request_id, doc_path, selfie_path)

    return {"id": request_id, "status": "PROCESSING", "message": "Verification started."}

@app.get("/status/{request_id}")
async def check_status(request_id: str):
    """Frontend calls this to see if AI has finished."""
    doc = await requests_collection.find_one({"_id": ObjectId(request_id)})
    if not doc: raise HTTPException(404, "Request not found.")
    return {"id": str(doc["_id"]), "status": doc["status"], "distance": doc.get("distance")}

# --- ADMIN SECTION ---

@app.get("/admin/pending")
async def get_pending():
    cursor = requests_collection.find({"status": "PENDING"}).sort("created_at", -1)
    return {"requests": [{**d, "_id": str(d["_id"])} async for d in cursor]}

@app.get("/admin/verified")
async def get_verified():
    cursor = requests_collection.find({"status": "APPROVED"}).sort("created_at", -1)
    return {"users": [{**d, "_id": str(d["_id"])} async for d in cursor]}

@app.post("/admin/action")
async def admin_decision(id: str = Form(...), action: str = Form(...)):
    new_status = "APPROVED" if action == "approve" else "REJECTED"
    await requests_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": new_status}})
    return {"status": "updated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)