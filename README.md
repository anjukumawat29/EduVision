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

## 🎮 Usage Guide

### For Students

#### 1. **Capture Face Photos**
1. Log in to student dashboard
2. Click **"Capture My Face"** button
3. Position face in camera frame
4. System captures 20 photos automatically
5. Photos saved to your gallery

#### 2. **Train Recognition Model**
1. Go to **"My Photos"** to verify you have enough photos
2. Click **"Train Model"** button on dashboard
3. Model training in progress (~10-30 seconds)
4. Success message when complete

#### 3. **View Attendance**
1. Check attendance stats on dashboard
2. View recent attendance records
3. Monitor personal attendance percentage

### For Teachers

#### 1. **Mark Attendance**
1. Click **"Mark Attendance"** button
2. Students appear in camera frame
3. Auto-marked when face recognized
4. Manual adjustments possible
5. Export to Excel when done

#### 2. **Monitor Behavior**
1. Go to **Behavior** page
2. Click **"Start Monitoring"** button
3. Select monitoring duration (default: 60 seconds)
4. Camera window opens with overlay
5. View real-time:
   - Student names with confidence
   - Behavior classification (attentive/distracted/phone)
   - Count of detected students
   - Timer countdown

#### 3. **Review Behavior Logs**
1. Check behavior alerts on dashboard
2. View detailed session logs
3. Identify patterns and trends

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

## ⚙️ Configuration

### Face Recognition Settings
Edit in `scan_behavior.py`:
```python
RECOGNITION_CONFIDENCE_THRESHOLD = 85  # Lower = stricter matching
```

### Head Pose Detection Settings
Edit in `scan_behavior.py`:
```python
YAW_THRESHOLD = 0.35      # Left/right rotation (0-1 scale)
PITCH_THRESHOLD = 0.35    # Up/down tilt (0-1 scale)
```

### Camera Settings
Edit in `camera_utils.py`:
```python
CAM_WIDTH = 960           # Camera resolution width
CAM_HEIGHT = 720          # Camera resolution height
WINDOW_WIDTH = 1100       # Display window width
WINDOW_HEIGHT = 750       # Display window height
```

### Behavior Monitoring Duration
Default: 60 seconds  
Override via command line:
```bash
python scan_behavior.py 120  # Monitor for 120 seconds
```

---

## 🐛 Known Issues & Solutions

### Issue #1: Photos Not Visible in Gallery
**Solution**: Ensure photos are saved to `/dataset/{username}/`
- Check: `capture_faces.py` uses absolute paths
- Verify: Files exist in correct directory

### Issue #2: Browser Caching Old Photos
**Solution**: Cache-busting with timestamps
- Templates include `?t={{ now|date:'U' }}` in image URLs
- Clear browser cache if needed

### Issue #3: Camera Window Not Displaying
**Solution**: macOS-specific OpenCV fix
- Uses TkAgg matplotlib backend
- Check: `macos_display_helper.py` is imported

### Issue #4: Face Recognition Model Not Found
**Solution**: Train model first
- Click "Train Model" button before behavior monitoring
- Requires minimum 20 photos per student

### Issue #5: YOLO Model Loading Slow
**Solution**: Normal behavior
- First run: 15-30 seconds (model downloads)
- Subsequent runs: <1 second (cached)

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

## 🚀 Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn core.wsgi --bind 0.0.0.0:8000 --workers 4
```

### Using Docker (Optional)
Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

Build and run:
```bash
docker build -t eduvision .
docker run -p 8000:8000 eduvision
```

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

## 👨‍💼 Author

**Anju Kumawat**  
GitHub: [@anjukumawat29](https://github.com/anjukumawat29)

---

## 🙏 Acknowledgments

- **YOLOv8** (Ultralytics) - Object detection
- **MediaPipe** (Google) - Pose estimation
- **OpenCV** - Computer vision library
- **Django** - Web framework

---

## 📞 Support

For issues, questions, or feedback:
1. Check existing GitHub Issues
2. Review `SYSTEM_DOCUMENTATION.md`
3. Open new GitHub Issue with details

---

## 🎯 Roadmap

### Planned Features
- [ ] Real-time SMS/Email alerts for unusual behavior
- [ ] Student engagement analytics dashboard
- [ ] Parent portal to view child's attendance
- [ ] ML model accuracy metrics and improvements
- [ ] Multi-camera support for large classrooms
- [ ] Mobile app for students
- [ ] Voice alerts during behavior monitoring
- [ ] Integration with school management system

### Performance Improvements
- [ ] Model compression for faster inference
- [ ] GPU acceleration support
- [ ] Redis caching for attendance queries
- [ ] Async task processing with Celery

---

**Last Updated**: April 3, 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
