from django.urls import path
from . import views
from .auth_views import (
    register_view, login_view, logout_view,
    student_dashboard, student_capture, student_delete_photo, student_photos, teacher_required,
)

def tr(view):
    return teacher_required(view)

urlpatterns = [
    # ── Auth (all public except logout) ────────────────────────────
    path("login/",    login_view,    name="login"),
    path("logout/",   logout_view,   name="logout"),
    path("register/", register_view, name="register"),

    # ── Student portal ──────────────────────────────────────────────
    path("student/",         student_dashboard, name="student_dashboard"),
    path("student/capture/", student_capture,   name="student_capture"),
    path("student/photos/",  student_photos,    name="student_photos"),
    path("student/photo/delete/<str:photo_name>/", student_delete_photo, name="student_delete_photo"),
    path("photo/<str:username>/<str:photo_name>/", views.student_photo, name="student_photo"),

    # ── Teacher views ───────────────────────────────────────────────
    path("",                            tr(views.home),             name="home"),
    path("students/",                   tr(views.student_list),     name="students"),
    path("students/delete/<str:name>/", tr(views.delete_student),   name="delete_student"),

    path("attendance/",         tr(views.attendance_page),   name="attendance_page"),
    path("attendance/scan/",    tr(views.mark_attendance),   name="mark_attendance"),
    path("attendance/export/",  tr(views.export_attendance), name="export_attendance"),

    path("bulk-train/", tr(views.bulk_train), name="bulk_train"),
    path("retrain/",    tr(views.retrain),    name="retrain"),
]