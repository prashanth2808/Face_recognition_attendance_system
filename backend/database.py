import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Connect to Supabase ──────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Students ─────────────────────────────────────

def save_student(name, roll_no, email, department, year, embedding):
    """Save a new student with their face embedding."""
    data = {
        "name":       name,
        "roll_no":    roll_no,
        "email":      email,
        "department": department,
        "year":       year,
        "embedding":  embedding   # list of floats
    }
    result = supabase.table("students").insert(data).execute()
    return result.data


def get_all_students():
    """Fetch all students with their embeddings for face matching."""
    result = supabase.table("students").select("*").execute()
    return result.data


def student_exists(roll_no):
    """Check if a student is already registered."""
    result = supabase.table("students").select("id").eq("roll_no", roll_no).execute()
    return len(result.data) > 0


# ── Attendance ────────────────────────────────────

def mark_attendance(student_id, name, roll_no, subject, period):
    """Mark attendance for a recognized student."""
    data = {
        "student_id": student_id,
        "name":       name,
        "roll_no":    roll_no,
        "subject":    subject,
        "period":     period
    }
    result = supabase.table("attendance").insert(data).execute()
    return result.data


def already_marked(roll_no, subject, period):
    """Prevent duplicate attendance in the same session."""
    from datetime import date
    today = date.today().isoformat()

    result = (
        supabase.table("attendance")
        .select("id")
        .eq("roll_no",  roll_no)
        .eq("subject",  subject)
        .eq("period",   period)
        .eq("date",     today)
        .execute()
    )
    return len(result.data) > 0


def get_attendance_report(subject=None):
    """Fetch attendance records, optionally filtered by subject."""
    query = supabase.table("attendance").select("*").order("date", desc=True)
    if subject:
        query = query.eq("subject", subject)
    result = query.execute()
    return result.data