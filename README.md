# FaceAttend - Smart Attendance System

FaceAttend is a modern, smart attendance system that uses state-of-the-art face recognition and liveness detection to securely and quickly mark student attendance. 

## Features

*   **Instant Face Recognition**: Uses the Facenet model (via `deepface`) to recognize faces in under 2 seconds.
*   **Liveness Detection**: Integrated MediaPipe face mesh on the frontend to detect natural eye blinking, preventing spoofing with static photos.
*   **Tamper-Proof**: No proxy attendance possible. The system uniquely identifies each student.
*   **Automated Reports**: Generates attendance records stored securely in a Supabase database.
*   **Beautiful UI**: A responsive, modern frontend designed with custom CSS.

## Tech Stack

### Frontend
*   HTML5, CSS3, Vanilla JavaScript
*   [MediaPipe](https://google.github.io/mediapipe/) (Face Mesh for Liveness/Blink Detection)
*   WebRTC (Camera access)

### Backend
*   [FastAPI](https://fastapi.tiangolo.com/) (High-performance Python web framework)
*   [DeepFace](https://github.com/serengil/deepface) (Facenet model for face embeddings)
*   [OpenCV](https://opencv.org/) (Face detection)
*   [Supabase](https://supabase.com/) (PostgreSQL Database as a Service)

## Project Structure

```
Face_attendance/
├── backend/
│   ├── main.py           # FastAPI application and routes
│   ├── face_engine.py    # DeepFace integration for embeddings and matching
│   ├── database.py       # Supabase database connection and queries
│   └── .env              # Environment variables (Supabase credentials)
├── frontend/
│   ├── index.html        # Landing page
│   ├── register.html     # Student registration with camera and liveness check
│   ├── attendance.html   # Attendance marking interface
│   └── style.css         # UI Styling
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### Prerequisites
*   Python 3.8+
*   A Supabase project (for database)

### 1. Backend Setup

1.  Navigate to the project directory:
    ```bash
    cd Face_attendance
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file inside the `backend/` directory and add your Supabase credentials:
    ```env
    SUPABASE_URL=your_supabase_project_url
    SUPABASE_KEY=your_supabase_anon_key
    ```
4.  Run the FastAPI server:
    ```bash
    cd backend
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

1.  You can serve the `frontend/` directory using any static file server. For example, using Python:
    ```bash
    cd frontend
    python -m http.server 5500
    ```
2.  Open your browser and navigate to `http://localhost:5500/index.html`.

## How it Works

1.  **Registration**: A student enters their details and turns on the camera. The frontend verifies they are a live person by asking them to blink twice. Once liveness is confirmed, it captures 3 photos. These photos are sent to the backend, which extracts face embeddings using DeepFace, averages them, and saves the student profile to Supabase.
2.  **Attendance**: The student looks at the camera on the attendance page. A photo is sent to the backend, which extracts the face embedding and compares it against all registered students using Euclidean distance. If a match is found within the threshold, attendance is marked in the database.
