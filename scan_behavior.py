import os
import cv2
import sys
import time
import json
import numpy as np
from ultralytics import YOLO

# ── macOS GUI fix ─────────────────────────────────────────────────
# Ensure OpenCV can display windows on macOS from subprocess
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for macOS compatibility
except:
    pass

# Import display helper
try:
    from macos_display_helper import enable_macos_display, safe_imshow, safe_waitkey, safe_destroyall
    enable_macos_display()
except:
    # Fallback if helper not available
    def safe_imshow(w, i):
        try:
            cv2.imshow(w, i)
        except:
            pass
    def safe_waitkey(d=1):
        try:
            return (cv2.waitKey(d) & 0xFF) == ord('q')
        except:
            return False
    def safe_destroyall():
        try:
            cv2.destroyAllWindows()
        except:
            pass

# ── MediaPipe (head pose) ─────────────────────────────────────────
try:
    import mediapipe as mp
    _mp_face  = mp.solutions.face_mesh
    _face_mesh = _mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    MEDIAPIPE_OK = True
    print("[behavior] MediaPipe loaded — head pose active")
except ImportError:
    MEDIAPIPE_OK = False
    print("[behavior] MediaPipe not installed — using Haar fallback for facing check")
    print("[behavior] Install with:  pip install mediapipe")

# ── Haar (fallback face detector if no mediapipe) ─────────────────
_cascade    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar        = cv2.CascadeClassifier(_cascade)

# ── Config ────────────────────────────────────────────────────────
duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# How far the head can yaw/pitch before we call it "looking away".
# 0 = perfectly centred, 1 = fully sideways. 0.35 is a comfortable threshold.
YAW_THRESHOLD   = 0.35
PITCH_THRESHOLD = 0.35

print("[behavior] Loading YOLO model... (this takes ~15 seconds on first run)")
yolo = YOLO("yolov8n.pt")
print("[behavior] YOLO model loaded")

# ── Load face recognizer for student identification ────────────────
recognizer = None
label_map = None
RECOGNITION_CONFIDENCE_THRESHOLD = 85

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_file = os.path.join(os.path.dirname(__file__), "ml_models", "lbph_model.yml")
    labels_file = os.path.join(os.path.dirname(__file__), "ml_models", "labels.pkl")
    
    if os.path.exists(model_file) and os.path.exists(labels_file):
        import pickle
        recognizer.read(model_file)
        with open(labels_file, "rb") as f:
            label_map = pickle.load(f)
        print("[behavior] Face recognition model loaded")
    else:
        recognizer = None
        print("[behavior] Model files not found - running without student identification")
except Exception as e:
    recognizer = None
    print(f"[behavior] Could not load recognizer: {e}")

# ── Haar Cascade for face detection ──────────────────────────────
_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(_cascade)

BEHAVIOR_COLORS = {
    "attentive":   (52,  211, 153),   # green
    "using phone": (68,   68, 239),   # red/blue
    "distracted":  (11,  158, 245),   # amber
}


# ── Head pose via MediaPipe ───────────────────────────────────────
def is_facing_camera_mediapipe(frame):
    """
    Returns True if the person is roughly facing the camera.
    Uses MediaPipe Face Mesh to estimate yaw + pitch from 3D landmarks.

    Key landmarks used:
      1   = nose tip
      33  = left eye outer corner
      263 = right eye outer corner
      152 = chin
      10  = forehead top

    Yaw  (left/right turn) — compare nose x to eye midpoint x
    Pitch (up/down tilt)   — compare nose y to eye-chin midpoint y
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return False   # no face found → looking away

    lm = res.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    nose        = pt(1)
    left_eye    = pt(33)
    right_eye   = pt(263)
    chin        = pt(152)
    forehead    = pt(10)

    eye_mid     = (left_eye + right_eye) / 2
    face_width  = np.linalg.norm(left_eye - right_eye) + 1e-6
    face_height = np.linalg.norm(forehead - chin)      + 1e-6

    # Normalised yaw: how far nose is horizontally from eye midpoint
    yaw   = abs(nose[0] - eye_mid[0]) / face_width

    # Normalised pitch: how far nose is vertically from eye-chin midpoint
    vert_mid = (eye_mid[1] + chin[1]) / 2
    pitch    = abs(nose[1] - vert_mid) / face_height

    return yaw < YAW_THRESHOLD and pitch < PITCH_THRESHOLD


def is_facing_camera_haar(frame):
    """
    Fallback when MediaPipe is not installed.
    Haar frontal-face cascade only detects faces that are roughly facing
    the camera — if no face is found, the person is likely turned away.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60)
    )
    return len(faces) > 0


def is_facing_camera(frame):
    if MEDIAPIPE_OK:
        return is_facing_camera_mediapipe(frame)
    return is_facing_camera_haar(frame)


# ── Face recognition ────────────────────────────────────────────────
def identify_student(frame):
    """
    Attempt to identify the student in the frame using LBPH recognizer.
    Returns (student_name, confidence) or (None, 0) if not recognized.
    Handles single student (largest face).
    """
    if recognizer is None or label_map is None:
        return None, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60)
    )
    
    if len(faces) == 0:
        return None, 0
    
    # Get largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_crop = gray[y:y+h, x:x+w]
    
    try:
        face_crop = cv2.resize(face_crop, (200, 200))
        label, confidence = recognizer.predict(face_crop)
        
        # Convert confidence to percentage (0-100)
        # Lower confidence value = better match
        recognition_score = 100 - confidence
        
        if confidence < RECOGNITION_CONFIDENCE_THRESHOLD and label in label_map:
            student_name = label_map[label]
            return student_name, recognition_score
    except:
        pass
    
    return None, 0


def identify_all_students(frame):
    """
    Identify ALL students/faces in the frame.
    Returns list of (x, y, w, h, student_name, confidence) for all detected faces.
    """
    if recognizer is None or label_map is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60)
    )
    
    identified_faces = []
    
    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        
        try:
            face_crop = cv2.resize(face_crop, (200, 200))
            label, confidence = recognizer.predict(face_crop)
            recognition_score = 100 - confidence
            
            if confidence < RECOGNITION_CONFIDENCE_THRESHOLD and label in label_map:
                student_name = label_map[label]
                identified_faces.append((x, y, w, h, student_name, recognition_score))
        except:
            pass
    
    return identified_faces


# ── Behavior classifier ───────────────────────────────────────────
def classify_behavior(detected, frame):
    """
    Priority order:
      1. Phone visible                    → using phone
      2. No person detected by YOLO       → distracted
      3. Person visible but NOT facing    → distracted
      4. Person facing + study material   → attentive
      5. Person facing, nothing else      → attentive
    
    Also identifies the student performing the activity.
    """
    # Try to identify the student
    student_name, conf = identify_student(frame)
    student_info = f" [{student_name}]" if student_name else ""
    
    if "cell phone" in detected:
        return "using phone", f"phone in hand{student_info}"

    if "person" not in detected:
        return "distracted", f"no person in frame"

    facing = is_facing_camera(frame)

    if not facing:
        return "distracted", f"looking away{student_info}"

    if any(c in detected for c in {"book", "laptop", "tv", "monitor"}):
        return "attentive", f"studying{student_info}"

    return "attentive", f"facing forward{student_info}"


# ── Overlay ───────────────────────────────────────────────────────
def draw_overlay(frame, results, behavior, reason, elapsed, total_dur):
    color = BEHAVIOR_COLORS.get(behavior, (180, 180, 180))
    
    # Identify ALL students in frame
    identified_faces = identify_all_students(frame)
    
    # Draw bounding boxes with student names for each face
    for (x, y, w, h, student_name, conf) in identified_faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw name label above face
        label = f"{student_name} ({conf:.0f}%)"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - label_h - 8), (x + label_w + 6, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x + 3, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for box in results.boxes:
        cls_name = yolo.names[int(box.cls[0])]
        if cls_name not in {"person", "cell phone", "book", "laptop", "tv"}:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = f"{cls_name} {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Top HUD with overall behavior
    hud_height = 72 if identified_faces else 72
    cv2.rectangle(frame, (0, 0), (frame.shape[1], hud_height), (15, 23, 42), -1)
    cv2.putText(frame, f"Behavior: {behavior.upper()}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, reason,
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1)
    
    # Show student count if any identified
    if identified_faces:
        num_students = len(identified_faces)
        student_text = f"Students detected: {num_students}"
        cv2.putText(frame, student_text,
                   (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Timer
    remaining  = max(0, total_dur - int(elapsed))
    timer_txt  = f"{remaining}s left"
    (tw, _), _ = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, timer_txt,
                (frame.shape[1] - tw - 12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
    cv2.putText(frame, "Q = stop early",
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)


# ── Camera open ───────────────────────────────────────────────────
# Create window FIRST on main thread
try:
    cv2.namedWindow("EduVision -- Behavior Monitor", cv2.WINDOW_NORMAL)
    print("[behavior] Window created")
except Exception as e:
    print(f"[behavior] Window creation warning: {e}")

# Now open camera
cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Prevent lag

print("[behavior] Starting camera...")
# Minimal warmup - just 5 frames instead of 40
for _ in range(5):
    cam.read()

# ── Monitor loop ──────────────────────────────────────────────────
log   = []
start = time.time()
print(f"[behavior] Starting {duration}s session...")

while True:
    elapsed = time.time() - start
    if elapsed >= duration:
        break

    ret, frame = cam.read()
    if not ret or frame is None:
        time.sleep(0.03)
        continue

    results  = yolo(frame, verbose=False)[0]
    detected = {yolo.names[int(b.cls[0])] for b in results.boxes}

    behavior, reason = classify_behavior(detected, frame)
    
    # Extract student identification for logging
    student_name, student_conf = identify_student(frame)

    draw_overlay(frame, results, behavior, reason, elapsed, duration)
    
    # Display window with macOS compatibility
    safe_imshow("EduVision -- Behavior Monitor", frame)

    if not log or elapsed - log[-1]["time"] >= 1.0:
        log.append({
            "time":     round(elapsed, 1),
            "behavior": behavior,
            "reason":   reason,
            "student":  student_name or "Unknown",
            "confidence": round(student_conf, 1) if student_name else 0,
            "objects":  list(detected),
        })

    if safe_waitkey(30):
        print("[behavior] Stopped early")
        break

cam.release()
safe_destroyall()

# ── Summary ───────────────────────────────────────────────────────
total       = len(log) or 1
attentive   = sum(1 for e in log if e["behavior"] == "attentive")
using_phone = sum(1 for e in log if e["behavior"] == "using phone")
distracted  = sum(1 for e in log if e["behavior"] == "distracted")

summary = {
    "entries":        log,
    "attentive":      attentive,
    "using_phone":    using_phone,
    "distracted":     distracted,
    "total":          total,
    "attentive_pct":  round(attentive   / total * 100),
    "phone_pct":      round(using_phone / total * 100),
    "distracted_pct": round(distracted  / total * 100),
}

print("RESULT:" + json.dumps(summary))