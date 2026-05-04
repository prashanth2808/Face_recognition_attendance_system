import numpy as np
import tempfile
import os
from deepface import DeepFace

# ── Settings ─────────────────────────────────────
MODEL_NAME  = "DeepFace"   # ← CHANGED (was "Facenet") — lighter, ~50MB RAM
THRESHOLD   = 20.0         # ← CHANGED (was 10.0) — DeepFace uses different scale
DETECTOR    = "opencv"

# ── Get Embedding from Image ──────────────────────
def get_embedding(image_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = DeepFace.represent(
            img_path          = tmp_path,
            model_name        = MODEL_NAME,
            detector_backend  = DETECTOR,
            enforce_detection = True
        )
        embedding = result[0]["embedding"]
        return embedding
    except Exception as e:
        raise ValueError(f"No face detected in image. Please try again. ({str(e)})")
    finally:
        os.remove(tmp_path)

# ── Average Multiple Embeddings ───────────────────
def average_embeddings(embeddings: list) -> list:
    arr = np.array(embeddings)
    avg = np.mean(arr, axis=0)
    return avg.tolist()

# ── Match Face Against All Students ───────────────
def find_match(live_embedding: list, all_students: list):
    live = np.array(live_embedding)
    best_match    = None
    best_distance = float("inf")

    for student in all_students:
        stored = np.array(student["embedding"])
        distance = np.linalg.norm(live - stored)
        if distance < best_distance:
            best_distance = distance
            best_match    = student

    if best_distance <= THRESHOLD:
        print(f"✅ Match found: {best_match['name']} (distance: {best_distance:.2f})")
        return best_match
    else:
        print(f"❌ No match. Closest distance: {best_distance:.2f}")
        return None
