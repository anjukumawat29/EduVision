# 🎓 EduVision - AI-Powered Student Attendance & Behavior Monitoring System

[![Python](https://img.shields.io/badge/Python-3.9.6-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2.29-green.svg)](https://www.djangoproject.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent classroom management system that uses **facial recognition** and **real-time behavior analysis** to automate attendance marking and monitor student behavior during class sessions.

---

## 🌟 Features

### 👥 **Facial Recognition Attendance**
- Automatic attendance marking using LBPH face recognition
- Multi-student simultaneous detection
- Confidence-based verification (85% threshold)
- Manual override for corrections

### 🎯 **Behavior Monitoring**
- Real-time student behavior classification:
  - **Attentive**: Facing camera, no distractions
  - **Distracted**: Looking at books, laptops, or TV
  - **Using Phone**: Detected cell phone in frame
- Identifies multiple students simultaneously during monitoring
- Head pose detection via MediaPipe (with Haar Cascade fallback)
- Session-based behavior logging

### 📸 **Face Photo Management**
- Students capture their own face photos for model training
- Photo gallery with delete functionality
- Minimum 20 photos per student for training
- Automatic preprocessing (200x200 grayscale)

### 📊 **Dashboard & Reporting**
- Teacher dashboard with behavior alerts
- Student dashboard with attendance stats
- Attendance export to Excel
- Behavior session logs and statistics

### 🔐 **Role-Based Access Control**
- **Students**: Photo capture, model training, view own attendance
- **Teachers**: Attendance marking, behavior monitoring, class management

---

## 📋 Prerequisites

- **Python**: 3.9.6+
- **macOS** (tested on macOS with AVFoundation camera support)
- **Git**: For version control
- **Camera**: Webcam or built-in camera

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anjukumawat29/EduVision.git
cd student-management
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Apply Database Migrations
```bash
python manage.py migrate
```

### 5. Create Superuser (Teacher Account)
```bash
python manage.py createsuperuser
# Follow prompts to create admin account
```

### 6. Collect Static Files (Optional for Development)
```bash
python manage.py collectstatic --noinput
```

---

## 🏃 Running the Application

### Start Django Development Server
```bash
python manage.py runserver
```

Then open your browser to:
```
http://localhost:8000
```

### Default Accounts
- **Admin URL**: `http://localhost:8000/admin`
- Superuser credentials: Use what you created with `createsuperuser`

---

## 📁 Project Structure

```
EduVision/
├── attendance/                 # Attendance & authentication app
│   ├── auth_views.py          # Student portal views
│   ├── views.py               # Teacher views & training
│   ├── models.py              # User, UserProfile, Attendance models
│   ├── urls.py                # Route definitions
│   └── dataset/               # Photo storage per student
│
├── behavior/                   # Behavior monitoring app
│   ├── views.py               # Behavior page & monitoring
│   ├── models.py              # BehaviorLog model
│   └── urls.py                # Behavior routes
│
├── core/                       # Django project settings
│   ├── settings.py            # Configuration
│   ├── urls.py                # Main URL router
│   ├── wsgi.py                # WSGI entry point
│   └── asgi.py                # ASGI entry point
│
├── dataset/                    # 📸 Face photos (ROOT LEVEL)
│   ├── username1/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── username2/
│
├── ml_models/                  # 🤖 Trained models
│   ├── lbph_model.yml         # LBPH face recognizer
│   └── labels.pkl             # Student name mappings
│
├── templates/                  # HTML templates
│   ├── base.html              # Base layout with sidebar
│   ├── home.html              # Teacher dashboard
│   ├── student_dashboard.html  # Student portal
│   ├── student_photos.html    # Photo gallery
│   ├── behavior.html          # Behavior monitoring page
│   ├── attendance.html        # Attendance marking
│   ├── login.html             # Authentication
│   └── register.html          # Registration
│
├── static/                     # CSS, JavaScript, images
│
├── scan_behavior.py           # ⭐ Behavior monitoring (subprocess)
├── capture_faces.py           # ⭐ Face photo capture (subprocess)
├── scan_attendance.py         # Attendance marking with face recognition
├── camera_utils.py            # Camera setup helpers
├── macos_display_helper.py    # macOS OpenCV display fixes
├── requirements.txt           # Python dependencies
├── manage.py                  # Django CLI
├── db.sqlite3                 # SQLite database
└── README.md                  # This file
```


---

## 🔑 Key Components

### Face Recognition Pipeline

```
Training Phase:
  Capture → Preprocess → Train LBPH → Save Model

Inference Phase:
  Frame → Detect Faces → Recognize → Display Results
```

**LBPH Recognizer**:
- Confidence threshold: **85**
  - Values below 85 = Recognized student
  - Values 85+ = Unknown/rejected
- Model file: `/ml_models/lbph_model.yml`
- Labels file: `/ml_models/labels.pkl`

### Behavior Classification

**Detects**:
- Person detection via YOLOv8
- Cell phone, books, laptops, TVs via YOLO
- Head pose (facing camera) via MediaPipe

**Classifications**:
- 🟢 **Attentive**: Facing camera + no distractions
- 🟡 **Distracted**: Facing camera + object detected
- 🔴 **Using Phone**: Cell phone detected in frame

### Head Pose Detection

**MediaPipe Approach** (Primary):
- Uses 3D facial landmarks
- Yaw threshold: 0.35 (head rotation)
- Pitch threshold: 0.35 (head tilt)

**Haar Cascade Fallback** (If MediaPipe unavailable):
- Detects frontal faces
- If detected = facing camera
- If not detected = looking away

---

## 📊 Performance Metrics

| Component | Time | Notes |
|-----------|------|-------|
| YOLO Load | 15-30s | First run only (cached after) |
| Face Recognizer Load | <100ms | Pickle file format |
| Camera Startup | ~0.5s | 5-frame warmup |
| Face Detection | ~50ms | Per frame (Haar Cascade) |
| Face Recognition | ~100ms | Per face (LBPH) |
| Behavior Classification | ~50ms | Per frame (YOLO) |

---

## 🔐 Security Notes

- **Passwords**: Hashed with Django's default PBKDF2
- **Photos**: Stored in `/dataset/` directory (not version controlled)
- **Models**: LBPH recognizer saved as binary file
- **Database**: SQLite (suitable for single-instance deployment)

**For Production**:
- Use environment variables for `SECRET_KEY`
- Switch to PostgreSQL
- Enable HTTPS
- Configure CORS properly
- Use gunicorn with reverse proxy (nginx)

---

## 📚 Dependencies

### Core Libraries
- **Django 4.2.29** - Web framework
- **OpenCV 4.8.0** - Computer vision
- **YOLOv8** - Object detection
- **MediaPipe** - Head pose estimation
- **NumPy** - Array operations
- **Pillow** - Image processing
- **openpyxl** - Excel file handling

See `requirements.txt` for complete list.

---

## 📖 Documentation

For detailed technical documentation, see:
- **`SYSTEM_DOCUMENTATION.md`** - Complete system reference
  - All functions documented
  - All bugs and solutions
  - Face recognition pipeline
  - Configuration reference

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

