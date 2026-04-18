import os
import sys
import json
from functools import wraps

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect


# ── Helper ──────────────────────────────────────────────────────────
def is_teacher(user):
    try:
        return bool(user.userprofile.is_teacher)
    except Exception:
        return False


# ── Decorator ───────────────────────────────────────────────────────
def teacher_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"/login/?next={request.path}")
        if not is_teacher(request.user):
            return redirect("student_dashboard")
        return view_func(request, *args, **kwargs)
    return wrapper


# ── Register ─────────────────────────────────────────────────────────
def register_view(request):
    if request.user.is_authenticated:
        return redirect_by_role(request.user)

    if request.method != "POST":
        return render(request, "register.html")

    name     = request.POST.get("name",     "").strip()
    password = request.POST.get("password", "").strip()
    role     = request.POST.get("role",     "student")

    errors = {}
    if not name:
        errors["name"] = "Name is required."
    if not password:
        errors["password"] = "Password is required."
    elif len(password) < 4:
        errors["password"] = "Password must be at least 4 characters."

    if errors:
        return render(request, "register.html", {"errors": errors, "post": request.POST})

    if User.objects.filter(username__iexact=name).exists():
        return render(request, "register.html", {
            "errors": {"name": f"Username '{name}' is already taken."},
            "post": request.POST,
        })

    user = User.objects.create_user(username=name, password=password)
    user.userprofile.is_teacher = (role == "teacher")
    user.userprofile.save()

    # NOTE: Camera/face capture is intentionally NOT launched at registration.
    # Students log in and click "Capture My Face" from their own dashboard.
    if role == "student":
        messages.success(request,
            f"Account created for '{name}'! Sign in and use 'Capture My Face' on your dashboard to register your face.")
    else:
        messages.success(request, f"Teacher account created for '{name}'. Please sign in.")
    return redirect("login")


# ── Login ────────────────────────────────────────────────────────────
def login_view(request):
    if request.user.is_authenticated:
        return redirect_by_role(request.user)

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            next_url = request.POST.get("next") or request.GET.get("next")
            if next_url and next_url not in ("/login/", "/register/"):
                return redirect(next_url)
            return redirect_by_role(user)
        return render(request, "login.html", {"error": True, "username": username})

    return render(request, "login.html")


# ── Logout ───────────────────────────────────────────────────────────
def logout_view(request):
    logout(request)
    return redirect("login")


def redirect_by_role(user):
    if is_teacher(user):
        return redirect("home")
    return redirect("student_dashboard")


# ── Student Dashboard ─────────────────────────────────────────────────
@login_required(login_url="/login/")
def student_dashboard(request):
    if is_teacher(request.user):
        return redirect("home")

    import openpyxl
    from django.conf import settings
    
    username = request.user.username
    
    # Check if student has photos in dataset
    from .views import get_students
    student_registered = username in get_students()
    
    # Read attendance from Excel file
    records = []
    total_classes = 0
    present_count = 0
    
    excel_file = os.path.join(settings.BASE_DIR, "attendance.xlsx")
    if os.path.exists(excel_file):
        try:
            wb = openpyxl.load_workbook(excel_file)
            ws = wb.active
            
            # Read all rows (skip header)
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not any(row):  # Skip empty rows
                    continue
                    
                name, date, time, status = row[0], row[1], row[2], row[3]
                
                # Match by name (case-insensitive)
                if name and str(name).lower() == username.lower():
                    records.append({
                        "name": name,
                        "date": date,
                        "time": time,
                        "status": status,
                        "subject": "General"
                    })
                    total_classes += 1
                    if status == "Present":
                        present_count += 1
        except Exception as e:
            print(f"[dashboard] Error reading Excel: {e}")
    
    # Calculate attendance percentage
    attendance_pct = round(present_count / total_classes * 100) if total_classes else 0
    
    # Reverse to show newest first
    records = records[::-1]

    # Get list of captured photos
    photos = []
    if student_registered:
        dataset_dir = os.path.join(settings.BASE_DIR, "dataset", username)
        if os.path.exists(dataset_dir):
            for filename in sorted(os.listdir(dataset_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    photos.append({
                        'name': filename,
                        'path': f'/media/dataset/{username}/{filename}'
                    })

    return render(request, "student_dashboard.html", {
        "records":        records,
        "total_classes":  total_classes,
        "present_count":  present_count,
        "attendance_pct": attendance_pct,
        "student":        student_registered,
        "photos":         json.dumps([p['name'] for p in photos]),  # JSON array of photo names
    })


# ── Student: re-capture faces from dashboard ──────────────────────────
@login_required(login_url="/login/")
def student_capture(request):
    if is_teacher(request.user):
        return redirect("home")
    if request.method != "POST":
        return redirect("student_dashboard")

    name        = request.user.username
    count       = 20
    script_path = os.path.join(settings.BASE_DIR, "capture_faces.py")

    if not os.path.exists(script_path):
        messages.error(request, "capture_faces.py not found. Ask your teacher to check the setup.")
        return redirect("student_dashboard")

    try:
        subprocess.Popen([sys.executable, script_path, name, str(count)])
        messages.success(request,
            f"Camera launched! A window will open — look at the camera to capture {count} photos.")
    except Exception as e:
        messages.error(request, f"Failed to launch camera: {e}")

    return redirect("student_dashboard")


# ── Student: delete a captured photo ──────────────────────────────
@login_required(login_url="/login/")
def student_delete_photo(request, photo_name):
    """Delete a single captured photo"""
    if is_teacher(request.user):
        return redirect("home")

    username = request.user.username
    dataset_dir = os.path.join(settings.BASE_DIR, "dataset", username)
    photo_path = os.path.join(dataset_dir, photo_name)

    # Security: ensure the file is in the student's folder
    if not os.path.abspath(photo_path).startswith(os.path.abspath(dataset_dir)):
        messages.error(request, "Invalid photo path.")
        return redirect("student_photos")

    # Security: only allow image files
    if not photo_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        messages.error(request, "Invalid file type.")
        return redirect("student_photos")

    try:
        if os.path.exists(photo_path):
            os.remove(photo_path)
            messages.success(request, f"Photo '{photo_name}' deleted.")
        else:
            messages.warning(request, "Photo not found.")
    except Exception as e:
        messages.error(request, f"Failed to delete photo: {e}")

    return redirect("student_photos")


# ── Student: view all captured photos ─────────────────────────────
@login_required(login_url="/login/")
def student_photos(request):
    """Display all captured photos in a gallery"""
    if is_teacher(request.user):
        return redirect("home")

    username = request.user.username
    photos = []
    
    dataset_dir = os.path.join(settings.BASE_DIR, "dataset", username)
    if os.path.exists(dataset_dir):
        for filename in sorted(os.listdir(dataset_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                photos.append(filename)
    
    return render(request, "student_photos.html", {
        "photos": photos,
        "photo_count": len(photos),
    })