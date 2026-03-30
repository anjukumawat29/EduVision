import cv2
import os
import sys
from ultralytics import YOLO

# ── Args ─────────────────────────────
name = sys.argv[1]  # student name

SAVE_DIR = "dataset"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 20

student_dir = os.path.join(SAVE_DIR, name)
os.makedirs(student_dir, exist_ok=True)

# ── YOLO face model ──────────────────
yolo = YOLO("yolov8n-face.pt")

# ── Camera ───────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

print(f"[capture] Registering: {name}")

# warmup
for _ in range(20):
    cap.read()

saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()

    # YOLO face detection
    results = yolo(frame, verbose=False)[0]

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # skip small faces
        if (x2 - x1) < 60 or (y2 - y1) < 60:
            continue

        face = frame[y1:y2, x1:x2]

        try:
            face = cv2.resize(face, (200, 200))
        except:
            continue

        # save image
        file_path = os.path.join(student_dir, f"{saved}.jpg")
        cv2.imwrite(file_path, face)

        saved += 1

        # draw box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)

        # only capture one face per frame
        break

    # ── UI Overlay ─────────────────────
    h, w = display.shape[:2]

    # Title
    cv2.putText(display, f"Registering: {name}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Progress bar
    bar_w = 600
    bar_h = 15
    margin = 20

    x_start = (w - bar_w) // 2
    y1 = h - margin - bar_h
    y2 = h - margin

    filled = int(bar_w * saved / COUNT)

    # background
    cv2.rectangle(display, (x_start, y1), (x_start + bar_w, y2), (50,50,50), -1)

    # progress
    cv2.rectangle(display, (x_start, y1), (x_start + filled, y2), (52,211,153), -1)

    # text above bar
    cv2.putText(display, f"{saved}/{COUNT}",
                (x_start, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # hint
    cv2.putText(display, "Q = stop",
                (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

    # show window
    display = cv2.resize(display, (1280, 720))
    cv2.imshow("Register Face", display)

    # stop conditions
    if saved >= COUNT:
        print("[capture] Done")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[capture] Stopped early")
        break

cap.release()
cv2.destroyAllWindows()