import os
import shutil
import uvicorn
import asyncio
import cv2
import pytesseract
import re
import numpy as np
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from scipy.spatial import distance

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kyc")

# --- CONFIGURATION ---
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Ultimate Identity & Admin Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- DATABASE SETUP ---
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client.get_database("kyc_database")
requests_collection = db.get_collection("requests")

# Connect to main application database for user updates
main_db = client.get_database(os.getenv("MONGODB_DB_NAME", "hire_expert"))
users_collection = main_db.get_collection("customers") # Main user collection
notifications_collection = main_db.get_collection("notifications") #  Notifications

# Precision Boundaries - More lenient for better user experience
AUTO_PASS_THRESHOLD = 0.55  # Increased from 0.42 - more lenient for auto-approval
ADMIN_REVIEW_THRESHOLD = 0.75  # Increased from 0.65 - more faces go to admin review instead of rejection
DUPLICATE_FACE_THRESHOLD = 0.30  # Decreased from 0.35 - stricter duplicate detection 

# --- 1. ADVANCED OCR & EXPIRY LOGIC ---

EXPIRY_KEYWORDS = re.compile(r'expir|valid\s*(until|thru|to|till)|date\s*of\s*expir|exp\.?\s*date|d\.?o\.?e', re.IGNORECASE)
ISSUE_KEYWORDS = re.compile(r'issue|issuance|date\s*of\s*(issue|birth)|d\.?o\.?b|d\.?o\.?i', re.IGNORECASE)

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def _preprocess_for_ocr(img, scales=[1.5, 2.0, 2.5]):
    """Multi-scale, multi-threshold OCR preprocessing for maximum text extraction."""
    results = []
    for scale in scales:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Otsu binarisation
        results.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        # High-contrast
        results.append(cv2.convertScaleAbs(gray, alpha=1.8, beta=-40))
        # Adaptive threshold
        results.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        # CLAHE for poor-lighting IDs
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        results.append(clahe.apply(gray))
    return results

def _extract_all_text(path):
    """Extract text using multiple preprocessing pipelines and merge."""
    img = cv2.imread(path)
    if img is None:
        return ""
    preprocessed = _preprocess_for_ocr(img)
    all_text = []
    for proc in preprocessed:
        try:
            text = pytesseract.image_to_string(proc, config='--psm 6')
            all_text.append(text)
        except Exception:
            continue
    merged = "\n".join(all_text)
    logger.info(f"OCR extracted text length: {len(merged)} chars")
    return merged

def _parse_dates_from_text(text):
    """Extract dates from OCR text using multiple format patterns."""
    dates_found = []
    
    # Pattern 1: DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
    for m in re.finditer(r'(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})', text):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Try DD/MM/YYYY first, then MM/DD/YYYY if day > 12
        for d, mo in [(day, month), (month, day)]:
            if 1 <= mo <= 12 and 1 <= d <= 31 and 1950 <= year <= 2050:
                try:
                    dates_found.append({"date": datetime(year, mo, d), "pos": m.start(), "raw": m.group()})
                    break
                except ValueError:
                    continue
    
    # Pattern 2: YYYY/MM/DD or YYYY-MM-DD
    for m in re.finditer(r'(\d{4})[\.\/\-](\d{1,2})[\.\/\-](\d{1,2})', text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1950 <= year <= 2050 and 1 <= month <= 12 and 1 <= day <= 31:
            try:
                dates_found.append({"date": datetime(year, month, day), "pos": m.start(), "raw": m.group()})
            except ValueError:
                continue
    
    # Pattern 3: Text-based dates — "12 Jan 2028", "January 12, 2028"
    text_date_pattern = r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*(\d{4})'
    for m in re.finditer(text_date_pattern, text, re.IGNORECASE):
        day = int(m.group(1))
        month_name = m.group(2).lower()[:3]
        year = int(m.group(3))
        if month_name in MONTH_MAP and 1 <= day <= 31 and 1950 <= year <= 2050:
            try:
                dates_found.append({"date": datetime(year, MONTH_MAP[month_name], day), "pos": m.start(), "raw": m.group()})
            except ValueError:
                continue
    
    # Pattern 4: "Jan 12, 2028" or "January 2028"
    text_date_pattern2 = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s*(\d{4})'
    for m in re.finditer(text_date_pattern2, text, re.IGNORECASE):
        month_name = m.group(1).lower()[:3]
        day = int(m.group(2))
        year = int(m.group(3))
        if month_name in MONTH_MAP and 1 <= day <= 31 and 1950 <= year <= 2050:
            try:
                dates_found.append({"date": datetime(year, MONTH_MAP[month_name], day), "pos": m.start(), "raw": m.group()})
            except ValueError:
                continue
    
    # Deduplicate by date value
    seen = set()
    unique = []
    for d in dates_found:
        key = d["date"].strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique.append(d)
    
    return unique

def check_document_expiry(path):
    """
    Advanced Expiry Gate with keyword-anchored date detection.
    
    Strategy:
    1. Extract all text via multi-scale OCR
    2. Find all dates in the text
    3. Look for expiry-related keywords near each date
    4. If an expiry date is found and it's in the past → EXPIRED
    5. If no expiry keyword is found, use heuristic: the latest date on the card is likely the expiry
    6. If no dates found at all, or all logic is inconclusive → PASS (benefit of doubt)
    """
    try:
        text = _extract_all_text(path)
        if not text.strip():
            logger.warning("Expiry Gate: No text extracted, passing by default")
            return True
        
        dates = _parse_dates_from_text(text)
        if not dates:
            logger.info("Expiry Gate: No dates found in document, passing")
            return True
        
        now = datetime.now()
        lines = text.split('\n')
        
        # Strategy 1: Find expiry-keyword-anchored dates
        expiry_date = None
        for d in dates:
            # Check surrounding context (100 chars before the date position)
            context_start = max(0, d["pos"] - 100)
            context = text[context_start:d["pos"] + len(d["raw"]) + 20]
            
            if EXPIRY_KEYWORDS.search(context) and not ISSUE_KEYWORDS.search(context):
                expiry_date = d["date"]
                logger.info(f"Expiry Gate: Found keyword-anchored expiry date: {expiry_date.strftime('%Y-%m-%d')} (raw: '{d['raw']}', context: '{context.strip()[:60]}')")
                break
        
        # Strategy 2: If no keyword match, use heuristic — the LATEST date is likely expiry
        if expiry_date is None and len(dates) >= 2:
            sorted_dates = sorted(dates, key=lambda x: x["date"])
            candidate = sorted_dates[-1]["date"]  # latest date
            # Only use heuristic if the latest date is reasonably far from the earliest (not just birth + issue)
            earliest = sorted_dates[0]["date"]
            if (candidate - earliest).days > 365:  # At least 1 year gap suggests issue→expiry pattern
                expiry_date = candidate
                logger.info(f"Expiry Gate: Heuristic — latest date used as expiry: {expiry_date.strftime('%Y-%m-%d')}")
        
        # Strategy 3: Single date on card — check if it's a future date (likely expiry) or past (likely issue)
        if expiry_date is None and len(dates) == 1:
            single = dates[0]["date"]
            # If the single date is in the future, it's likely the expiry — pass
            # If it's in the past, it could be issue date — inconclusive, pass with benefit of doubt
            logger.info(f"Expiry Gate: Single date {single.strftime('%Y-%m-%d')} found, passing (inconclusive)")
            return True
        
        # Final decision
        if expiry_date is None:
            logger.info("Expiry Gate: No expiry date determined, passing")
            return True
        
        if expiry_date < now:
            logger.warning(f"Expiry Gate: EXPIRED — expiry date {expiry_date.strftime('%Y-%m-%d')} is in the past")
            return False
        
        logger.info(f"Expiry Gate: VALID — expiry date {expiry_date.strftime('%Y-%m-%d')} is in the future")
        return True
        
    except Exception as e:
        logger.error(f"Expiry Gate: Exception {e}, passing by default")
        return True

def extract_id_recursive(path):
    """
    Advanced Anchor-Based OCR: Multi-pipeline ID extraction with keyword proximity.
    
    Uses multiple preprocessing stages and looks for ID-like patterns near
    anchor keywords (Identity, Number, CNIC, etc.)
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return None, ""
        
        preprocessed = _preprocess_for_ocr(img, scales=[2.0, 2.5, 3.0])
        
        # ID patterns: CNIC (XXXXX-XXXXXXX-X), UAE (784-XXXX-XXXXXXX-X), Passport (alphanumeric 7-20)
        cnic_pattern = r'\b(\d{5}[\-\s]?\d{7}[\-\s]?\d{1})\b'
        generic_pattern = r'\b(?=.*\d)[A-Z0-9][\-A-Z0-9]{6,21}\b'
        id_anchors = re.compile(r'identity|number|cnic|nadra|nic|passport|id\s*(?:no|number|card)|document\s*(?:no|number)', re.IGNORECASE)
        
        noise_words = {"ISLAMIC", "PAKISTAN", "EUROPA", "REPUBLIC", "GOVERNMENT", "NATIONAL", 
                       "AUTHORITY", "REGISTRATION", "MINISTRY", "DEPARTMENT", "PROVINCE"}
        
        best_candidates = []
        all_text_merged = []
        
        for proc in preprocessed:
            try:
                d = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT)
                text_full = " ".join(d['text']).upper()
                all_text_merged.append(text_full)
                
                # Priority 1: CNIC-format numbers (highest confidence)
                for m in re.finditer(cnic_pattern, text_full):
                    val = m.group(1).replace(" ", "")
                    if len(re.sub(r'[^0-9]', '', val)) == 13:
                        best_candidates.append({"value": val, "priority": 1, "pos": m.start()})
                
                # Priority 2: Generic ID near anchor keywords
                for m in re.finditer(generic_pattern, text_full):
                    val = m.group()
                    if any(noise in val for noise in noise_words):
                        continue
                    if not any(c.isdigit() for c in val) or len(val) < 7:
                        continue
                    
                    context_start = max(0, m.start() - 80)
                    context = text_full[context_start:m.end() + 20]
                    priority = 2 if id_anchors.search(context) else 3
                    best_candidates.append({"value": val, "priority": priority, "pos": m.start()})
                    
            except Exception:
                continue
        
        merged_text = "\n".join(all_text_merged)
        
        if not best_candidates:
            logger.warning("ID Extraction: No candidates found")
            return None, merged_text
        
        # Sort by priority (1=best), then by position (earlier = more likely ID field)
        best_candidates.sort(key=lambda x: (x["priority"], x["pos"]))
        
        # Deduplicate: normalize and pick best
        seen_normalized = set()
        for c in best_candidates:
            normalized = re.sub(r'[^A-Z0-9]', '', c["value"])
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                logger.info(f"ID Extraction: Best match = '{c['value']}' (priority={c['priority']})")
                return c["value"], merged_text
        
        return None, merged_text
    except Exception as e:
        logger.error(f"ID Extraction error: {e}")
        return None, ""

# --- 2. BIOMETRIC ENSEMBLE ---

def get_super_embedding(path):
    """Structural + Textural pattern fusion with error handling."""
    try:
        logger.info(f"Generating embeddings for: {path}")

        # Check if file exists and is readable
        if not os.path.exists(path):
            raise Exception(f"Image file not found: {path}")

        # Verify image can be loaded
        test_img = cv2.imread(path)
        if test_img is None:
            raise Exception(f"Cannot load image: {path}")

        logger.info(f"Image loaded successfully: {test_img.shape}")

        # Generate embeddings with error handling
        emb_arc = DeepFace.represent(img_path=path, model_name="ArcFace", enforce_detection=False)
        if not emb_arc or len(emb_arc) == 0 or 'embedding' not in emb_arc[0]:
            raise Exception("ArcFace embedding generation failed")

        emb_vgg = DeepFace.represent(img_path=path, model_name="VGG-Face", enforce_detection=False)
        if not emb_vgg or len(emb_vgg) == 0 or 'embedding' not in emb_vgg[0]:
            raise Exception("VGG-Face embedding generation failed")

        arc_embedding = emb_arc[0]["embedding"]
        vgg_embedding = emb_vgg[0]["embedding"]

        logger.info(f"Embeddings generated - ArcFace: {len(arc_embedding)}, VGG-Face: {len(vgg_embedding)}")

        # Combine embeddings
        combined = np.array(arc_embedding + vgg_embedding)
        logger.info(f"Combined embedding length: {len(combined)}")

        return combined

    except Exception as e:
        logger.error(f"Error generating super embedding for {path}: {e}", exc_info=True)
        raise Exception(f"Face embedding generation failed: {str(e)}")

# --- 3. MAIN VERIFICATION TASK ---

async def run_kyc_task(req_id: str, doc_path: str, selfie_path: str, typed_id: str):
    try:
        logger.info(f"[{req_id}] Starting KYC verification pipeline")
        logger.info(f"[{req_id}] Document path: {doc_path}")
        logger.info(f"[{req_id}] Selfie path: {selfie_path}")
        logger.info(f"[{req_id}] Typed ID: {typed_id}")

        # Validate file existence and readability
        if not os.path.exists(doc_path):
            raise Exception(f"Document file not found: {doc_path}")

        if not os.path.exists(selfie_path):
            raise Exception(f"Selfie file not found: {selfie_path}")

        doc_size = os.path.getsize(doc_path)
        selfie_size = os.path.getsize(selfie_path)
        logger.info(f"[{req_id}] File sizes - Doc: {doc_size} bytes, Selfie: {selfie_size} bytes")

        if doc_size == 0 or selfie_size == 0:
            raise Exception("One or more uploaded files are empty")

        # Test image loading
        doc_img = cv2.imread(doc_path)
        if doc_img is None:
            raise Exception(f"Failed to load document image: {doc_path}")

        selfie_img = cv2.imread(selfie_path)
        if selfie_img is None:
            raise Exception(f"Failed to load selfie image: {selfie_path}")

        logger.info(f"[{req_id}] Images loaded successfully - Doc: {doc_img.shape}, Selfie: {selfie_img.shape}")

        # Step A: Expiry Gate
        logger.info(f"[{req_id}] Gate A: Checking document expiry...")
        try:
            if not check_document_expiry(doc_path):
                await requests_collection.update_one({"_id": ObjectId(req_id)},
                    {"$set": {"status": "REJECTED", "message": "ID Card Expired.", "gate_failed": "expiry"}})
                logger.warning(f"[{req_id}] REJECTED at Gate A: Expired document")
                return
            logger.info(f"[{req_id}] Gate A: PASSED - Document is valid")
        except Exception as e:
            logger.error(f"[{req_id}] Gate A error: {e}")
            # Continue with other gates even if expiry check fails

        # Step B: Smart ID Match Gate
        logger.info(f"[{req_id}] Gate B: Extracting & matching ID number...")
        try:
            extracted_id, ocr_text = extract_id_recursive(doc_path)
            clean_typed = re.sub(r'[^A-Z0-9]', '', typed_id.upper())
            clean_ext = re.sub(r'[^A-Z0-9]', '', extracted_id.upper()) if extracted_id else ""

            logger.info(f"[{req_id}] Gate B: Typed ID: '{typed_id}' -> '{clean_typed}'")
            logger.info(f"[{req_id}] Gate B: Extracted ID: '{extracted_id}' -> '{clean_ext}'")

            if not clean_ext:
                logger.warning(f"[{req_id}] Gate B: OCR couldn't extract ID, skipping gate (benefit of doubt)")
                # Don't fail here - let admin review if needed
            elif clean_typed != clean_ext:
                match_ratio = sum(1 for a, b in zip(clean_typed, clean_ext) if a == b) / max(len(clean_typed), len(clean_ext), 1)
                if match_ratio >= 0.80 and abs(len(clean_typed) - len(clean_ext)) <= 2:
                    logger.info(f"[{req_id}] Gate B: Fuzzy match accepted ({match_ratio:.0%})")
                else:
                    # Instead of rejecting, send to admin review for manual verification
                    await requests_collection.update_one({"_id": ObjectId(req_id)},
                        {"$set": {"status": "PENDING", "message": f"ID number requires manual verification. OCR extracted: {extracted_id}", "gate_failed": "id_match", "extracted_id": extracted_id, "ocr_text": ocr_text[:500]}})
                    logger.warning(f"[{req_id}] Gate B: ID mismatch - sending to admin review (match={match_ratio:.0%})")
                    return
            else:
                logger.info(f"[{req_id}] Gate B: Exact ID match confirmed")
        except Exception as e:
            logger.error(f"[{req_id}] Gate B error: {e}")
            # Continue with other gates

        # Step C: Biometric Duplicate Check
        logger.info(f"[{req_id}] Gate C: Checking for duplicate faces...")
        try:
            new_emb = get_super_embedding(selfie_path)
            logger.info(f"[{req_id}] Gate C: Generated embedding with {len(new_emb)} dimensions")

            cursor = requests_collection.find({"status": "APPROVED", "super_embedding": {"$exists": True}})
            duplicate_found = False

            async for user in cursor:
                try:
                    cos_dist = distance.cosine(new_emb, np.array(user["super_embedding"]))
                    if cos_dist < DUPLICATE_FACE_THRESHOLD:
                        await requests_collection.update_one({"_id": ObjectId(req_id)},
                            {"$set": {"status": "REJECTED", "message": "Face already registered.", "gate_failed": "duplicate"}})
                        logger.warning(f"[{req_id}] REJECTED at Gate C: Duplicate face (distance={cos_dist:.4f})")
                        duplicate_found = True
                        break
                except Exception as e:
                    logger.error(f"[{req_id}] Error checking duplicate against user {user.get('_id')}: {e}")
                    continue

            if duplicate_found:
                return

            logger.info(f"[{req_id}] Gate C: PASSED - No duplicate faces found")
        except Exception as e:
            logger.error(f"[{req_id}] Gate C error: {e}")
            # If embedding fails, we can't check duplicates, so continue

        # Step D: Deep Pattern Match
        logger.info(f"[{req_id}] Gate D: Running biometric face comparison...")
        try:
            # First, verify that faces can be detected in both images
            try:
                doc_faces = DeepFace.detectFace(img_path=doc_path, detector_backend='opencv')
                selfie_faces = DeepFace.detectFace(img_path=selfie_path, detector_backend='opencv')
                logger.info(f"[{req_id}] Face detection successful - Doc faces: {len(doc_faces) if isinstance(doc_faces, list) else 1}, Selfie faces: {len(selfie_faces) if isinstance(selfie_faces, list) else 1}")
            except Exception as face_detect_error:
                logger.warning(f"[{req_id}] Face detection failed: {face_detect_error}")
                await requests_collection.update_one({"_id": ObjectId(req_id)},
                    {"$set": {"status": "REJECTED", "message": "Could not detect faces in images. Please ensure clear, well-lit photos with visible faces.", "gate_failed": "face_detection"}})
                return

            res_arc = DeepFace.verify(doc_path, selfie_path, model_name="ArcFace", enforce_detection=False)
            res_vgg = DeepFace.verify(doc_path, selfie_path, model_name="VGG-Face", enforce_detection=False)

            avg_dist = (res_arc["distance"] * 0.7) + (res_vgg["distance"] * 0.3)
            logger.info(f"[{req_id}] Gate D: ArcFace={res_arc['distance']:.4f}, VGG={res_vgg['distance']:.4f}, Weighted={avg_dist:.4f}")

            # More detailed decision logic with better user feedback
            if avg_dist <= AUTO_PASS_THRESHOLD:
                status, msg = "APPROVED", "Biometric verification successful! Your identity has been confirmed."
                embedding_to_save = new_emb if 'new_emb' in locals() else None
            elif avg_dist <= ADMIN_REVIEW_THRESHOLD:
                status, msg = "PENDING", "Your verification requires manual review. We'll notify you within 24-48 hours."
                embedding_to_save = None
            else:
                # Provide specific feedback based on distance
                if avg_dist > 0.8:
                    msg = "Faces don't appear to match. Please ensure you're using a recent photo of yourself and that your ID document photo is clear."
                elif avg_dist > 0.7:
                    msg = "Face similarity is low. Try taking the selfie in better lighting with your face clearly visible."
                else:
                    msg = "Biometric verification failed. Please ensure your selfie matches your ID document photo."
                status = "REJECTED"
                embedding_to_save = None

            update_data = {
                "status": status,
                "distance": avg_dist,
                "id_number": typed_id,
                "message": msg,
                "arcface_distance": res_arc["distance"],
                "vggface_distance": res_vgg["distance"]
            }

            if embedding_to_save is not None:
                update_data["super_embedding"] = embedding_to_save

            await requests_collection.update_one({"_id": ObjectId(req_id)}, {"$set": update_data})
            logger.info(f"[{req_id}] Pipeline result: {status} — {msg}")

        except Exception as e:
            logger.error(f"[{req_id}] Gate D error: {e}", exc_info=True)
            error_msg = "Face verification processing failed. Please try again with clearer images."
            if "face" in str(e).lower():
                error_msg = "Could not process facial features. Please ensure both images show clear faces."
            await requests_collection.update_one({"_id": ObjectId(req_id)},
                {"$set": {"status": "ERROR", "message": error_msg}})

    except Exception as e:
        logger.error(f"[{req_id}] Pipeline EXCEPTION: {e}", exc_info=True)
        await requests_collection.update_one({"_id": ObjectId(req_id)},
            {"$set": {"status": "ERROR", "message": f"Processing error: {str(e)[:100]}"}})

# --- 4. ENDPOINTS ---

@app.post("/verify")
async def verify(bg: BackgroundTasks, id_number: str = Form(...), cnic: UploadFile = File(...), selfie: UploadFile = File(...), user_id: str = Form(None)):
    logger.info("=== KYC Verification Request Started ===")
    logger.info(f"Received form data - id_number: '{id_number}', cnic_filename: '{cnic.filename if cnic else None}', selfie_filename: '{selfie.filename if selfie else None}', user_id: '{user_id}'")
    
    try:
        # Validate inputs
        if not id_number or not id_number.strip():
            logger.error("Validation failed: ID number is required")
            raise HTTPException(status_code=400, detail="ID number is required")

        if not cnic or not cnic.filename:
            logger.error("Validation failed: CNIC document is required")
            raise HTTPException(status_code=400, detail="CNIC document is required")

        if not selfie or not selfie.filename:
            logger.error("Validation failed: Selfie image is required")
            raise HTTPException(status_code=400, detail="Selfie image is required")

        logger.info("Input validation passed")

        # Generate unique filenames with proper extensions
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cnic_ext = os.path.splitext(cnic.filename)[1].lower() or '.jpg'
        selfie_ext = os.path.splitext(selfie.filename)[1].lower() or '.jpg'

        # Ensure valid image extensions
        if cnic_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            cnic_ext = '.jpg'
        if selfie_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            selfie_ext = '.jpg'

        d_path = os.path.join(UPLOAD_DIR, f"d_{ts}{cnic_ext}")
        s_path = os.path.join(UPLOAD_DIR, f"s_{ts}{selfie_ext}")

        # Save uploaded files
        with open(d_path, "wb") as f:
            shutil.copyfileobj(cnic.file, f)

        with open(s_path, "wb") as f:
            shutil.copyfileobj(selfie.file, f)

        logger.info(f"Files saved - CNIC: {d_path} ({os.path.getsize(d_path)} bytes), Selfie: {s_path} ({os.path.getsize(s_path)} bytes)")

        # Verify files were saved and are readable
        if not os.path.exists(d_path) or os.path.getsize(d_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to save CNIC document")

        if not os.path.exists(s_path) or os.path.getsize(s_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to save selfie image")

        # Quick validation that files are valid images
        try:
            img_test = cv2.imread(d_path)
            if img_test is None:
                raise Exception("Invalid CNIC image format")
        except Exception as e:
            logger.error(f"CNIC image validation failed: {e}")
            # Clean up invalid files
            if os.path.exists(d_path):
                os.remove(d_path)
            if os.path.exists(s_path):
                os.remove(s_path)
            raise HTTPException(status_code=400, detail="CNIC document is not a valid image file")

        try:
            img_test = cv2.imread(s_path)
            if img_test is None:
                raise Exception("Invalid selfie image format")
        except Exception as e:
            logger.error(f"Selfie image validation failed: {e}")
            # Clean up invalid files
            if os.path.exists(d_path):
                os.remove(d_path)
            if os.path.exists(s_path):
                os.remove(s_path)
            raise HTTPException(status_code=400, detail="Selfie image is not a valid image file")

        # Create KYC request with user_id if provided
        request_data = {
            "status": "PROCESSING",
            "id_number": id_number.strip(),
            "cnic_path": d_path,
            "selfie_path": s_path,
            "created_at": datetime.utcnow(),
            "file_sizes": {
                "cnic": os.path.getsize(d_path),
                "selfie": os.path.getsize(s_path)
            }
        }
        if user_id:
            request_data["user_id"] = user_id

        res = await requests_collection.insert_one(request_data)
        logger.info(f"Created KYC request {res.inserted_id} for ID {id_number.strip()}")

        # Start background processing
        bg.add_task(run_kyc_task, str(res.inserted_id), d_path, s_path, id_number.strip())

        return {"id": str(res.inserted_id), "message": "Verification request submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /verify endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)[:100]}")

@app.get("/status/{id}")
async def check_status(id: str):
    doc = await requests_collection.find_one({"_id": ObjectId(id)})
    return {
        "status": doc["status"],
        "message": doc.get("message"),
        "id_number": doc.get("id_number"),
        "distance": doc.get("distance")
    }

# --- 5. ADMIN SECTION (Detailed View) ---

@app.get("/admin/pending")
async def get_pending():
    cursor = requests_collection.find({"status": "PENDING"})
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["id"] = doc["_id"]
        if doc.get("cnic_path"):
            doc["cnic_url"] = f"/static/uploads/{os.path.basename(doc['cnic_path'])}"
        if doc.get("selfie_path"):
            doc["selfie_url"] = f"/static/uploads/{os.path.basename(doc['selfie_path'])}"
        results.append(doc)
    return {"pending_requests": results}

@app.get("/admin/request/{id}")
async def get_request_details(id: str):
    """Fetches full profile and image URLs for the Admin."""
    doc = await requests_collection.find_one({"_id": ObjectId(id)})
    if not doc: raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": str(doc["_id"]), "status": doc.get("status"), "distance": doc.get("distance"),
        "id_number": doc.get("id_number"), "message": doc.get("message"),
        "cnic_url": f"/static/uploads/{os.path.basename(doc['cnic_path'])}",
        "selfie_url": f"/static/uploads/{os.path.basename(doc['selfie_path'])}"
    }

@app.get("/admin/user/{user_id}")
async def get_user_kyc(user_id: str):
    """Get the most recent KYC request for a user by their user_id."""
    doc = await requests_collection.find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)]
    )
    if not doc:
        raise HTTPException(status_code=404, detail="No KYC request found for this user")
    result = {
        "id": str(doc["_id"]),
        "status": doc.get("status"),
        "id_number": doc.get("id_number"),
        "message": doc.get("message"),
        "distance": doc.get("distance"),
    }
    if doc.get("cnic_path"):
        result["cnic_url"] = f"/static/uploads/{os.path.basename(doc['cnic_path'])}"
    if doc.get("selfie_path"):
        result["selfie_url"] = f"/static/uploads/{os.path.basename(doc['selfie_path'])}"
    return result

@app.post("/admin/action")
async def admin_action(id: str = Form(...), action: str = Form(...)):
    new_status = "APPROVED" if action == "approve" else "REJECTED"
    doc = await requests_collection.find_one({"_id": ObjectId(id)})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Update KYC request status
    if new_status == "APPROVED":
        emb = get_super_embedding(doc["selfie_path"]).tolist()
        await requests_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": new_status, "super_embedding": emb, "updated_at": datetime.utcnow()}}
        )
    else:
        await requests_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": new_status, "updated_at": datetime.utcnow()}}
        )
    
    # Update user verification status in main database if user_id exists
    if doc.get("user_id"):
        try:
            user_id = ObjectId(doc["user_id"])
            
            if new_status == "APPROVED":
                # Update user's verification status
                await users_collection.update_one(
                    {"_id": user_id},
                    {"$set": {
                        "isVerified": True,
                        "profile.verification.idCard.status": "verified",
                        "profile.verification.idCard.verifiedAt": datetime.utcnow(),
                        "profile.verification.faceDetection.status": "verified",
                        "profile.verification.faceDetection.verifiedAt": datetime.utcnow()
                    }}
                )
                
                # Create notification
                notification = {
                    "userId": user_id,
                    "type": "verification_approved",
                    "status": "approved",
                    "message": "🎉 Your identity verification has been approved! You now have a verified badge on your profile.",
                    "read": False,
                    "createdAt": datetime.utcnow(),
                    "updatedAt": datetime.utcnow()
                }
                await notifications_collection.insert_one(notification)
                logger.info(f"✅ User {user_id} verified and notified")
            else:
                # Rejected - update status
                await users_collection.update_one(
                    {"_id": user_id},
                    {"$set": {
                        "profile.verification.idCard.status": "rejected",
                        "profile.verification.faceDetection.status": "rejected"
                    }}
                )
                
                # Create rejection notification
                notification = {
                    "userId": user_id,
                    "type": "verification_rejected",
                    "status": "rejected",
                    "message": "Your identity verification was not approved. Please contact support if you believe this is an error.",
                    "read": False,
                    "createdAt": datetime.utcnow(),
                    "updatedAt": datetime.utcnow()
                }
                await notifications_collection.insert_one(notification)
                logger.info(f"❌ User {user_id} verification rejected and notified")
                
        except Exception as e:
            logger.error(f"Error updating user status: {e}")
            # Continue even if user update fails
    
    return {"status": "Updated", "user_updated": bool(doc.get("user_id"))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
