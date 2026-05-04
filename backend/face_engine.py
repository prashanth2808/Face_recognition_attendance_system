import numpy as np
import tempfile
import os
from deepface import DeepFace


# ── Settings ─────────────────────────────────────
MODEL_NAME      = "Facenet"   # accurate & fast
THRESHOLD       = 10.0        # lower = stricter matching
DETECTOR        = "opencv"    # face detector backend


# ── Get Embedding from Image ──────────────────────

def get_embedding(image_bytes: bytes) -> list:
    """
    Takes image as bytes, returns face embedding (list of numbers).
    Raises an error if no face is detected.
    """
    # Save bytes to a temp file (DeepFace needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = DeepFace.represent(
            img_path        = tmp_path,
            model_name      = MODEL_NAME,
            detector_backend = DETECTOR,
            enforce_detection = True
        )
        embedding = result[0]["embedding"]
        return embedding

    except Exception as e:
        raise ValueError(f"No face detected in image. Please try again. ({str(e)})")

    finally:
        os.remove(tmp_path)   # clean up temp file


# ── Average Multiple Embeddings ───────────────────

def average_embeddings(embeddings: list) -> list:
    """
    Takes a list of embeddings (from 3 photos) and
    returns one averaged embedding for better accuracy.
    """
    arr = np.array(embeddings)
    avg = np.mean(arr, axis=0)
    return avg.tolist()


# ── Match Face Against All Students ───────────────

def find_match(live_embedding: list, all_students: list):
    """
    Compares live face embedding against all stored student embeddings.
    Returns the best matching student or None if no match found.

    all_students → list of student dicts from database
    Each student must have an 'embedding' field.
    """
    live = np.array(live_embedding)

    best_match    = None
    best_distance = float("inf")

    for student in all_students:
        stored = np.array(student["embedding"])

        # Euclidean distance between two embeddings
        distance = np.linalg.norm(live - stored)

        if distance < best_distance:
            best_distance = distance
            best_match    = student

    # Only return match if distance is within threshold
    if best_distance <= THRESHOLD:
        print(f"✅ Match found: {best_match['name']} (distance: {best_distance:.2f})")
        return best_match
    else:
        print(f"❌ No match. Closest distance: {best_distance:.2f}")
        return None
