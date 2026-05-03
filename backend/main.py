import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv

from face_engine import get_embedding, average_embeddings, find_match
from database   import (
    save_student, get_all_students, student_exists,
    mark_attendance, already_marked, get_attendance_report
)

load_dotenv()

# ── App Setup ─────────────────────────────────────
app = FastAPI(title="FaceAttend API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your frontend URL when deployed
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ FaceAttend API is running!"}


# ── Register Student ──────────────────────────────
@app.post("/register")
async def register(
    name:       str        = Form(...),
    roll_no:    str        = Form(...),
    email:      str        = Form(...),
    department: str        = Form(...),
    year:       str        = Form(...),
    photos:     List[UploadFile] = File(...)
):
    # Check if already registered
    if student_exists(roll_no):
        raise HTTPException(400, f"Student with roll number {roll_no} is already registered.")

    # Check we have exactly 3 photos
    if len(photos) != 3:
        raise HTTPException(400, "Please upload exactly 3 face photos.")

    # Get embedding from each photo
    embeddings = []
    for i, photo in enumerate(photos):
        image_bytes = await photo.read()
        try:
            embedding = get_embedding(image_bytes)
            embeddings.append(embedding)
        except ValueError as e:
            raise HTTPException(400, f"Photo {i+1}: {str(e)}")

    # Average the 3 embeddings into one
    final_embedding = average_embeddings(embeddings)

    # Save to database
    result = save_student(name, roll_no, email, department, year, final_embedding)

    return {
        "success": True,
        "message": f"Student {name} registered successfully!",
        "data":    result
    }


# ── Mark Attendance ───────────────────────────────
@app.post("/attendance")
async def attendance(
    photo:   UploadFile = File(...),
    subject: str        = Form(...),
    period:  str        = Form(...)
):
    # Get live face embedding
    image_bytes = await photo.read()
    try:
        live_embedding = get_embedding(image_bytes)
    except ValueError as e:
        return {"recognized": False, "message": str(e)}

    # Fetch all students from DB
    all_students = get_all_students()
    if not all_students:
        return {"recognized": False, "message": "No students registered yet."}

    # Find matching student
    match = find_match(live_embedding, all_students)

    if not match:
        return {"recognized": False, "message": "Face not recognized."}

    # Check for duplicate attendance
    if already_marked(match["roll_no"], subject, period):
        return {
            "recognized": True,
            "name":       match["name"],
            "roll_no":    match["roll_no"],
            "message":    "Attendance already marked for this session.",
            "duplicate":  True
        }

    # Mark attendance in DB
    mark_attendance(match["id"], match["name"], match["roll_no"], subject, period)

    return {
        "recognized": True,
        "name":       match["name"],
        "roll_no":    match["roll_no"],
        "message":    "Attendance marked successfully!",
        "duplicate":  False
    }


# ── Get Attendance Report ─────────────────────────
@app.get("/report")
def report(subject: str = None):
    data = get_attendance_report(subject)
    return {"total": len(data), "records": data}