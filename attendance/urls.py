from django.urls import path
from . import views
 
urlpatterns = [
    path("",                             views.home,             name="home"),
    path("register/",                    views.register_student, name="register"),
    path("bulk-train/",                  views.bulk_train,       name="bulk_train"),
    path("retrain/",                     views.retrain,          name="retrain"),
    path("students/",                    views.student_list,     name="students"),
    path("students/delete/<str:name>/",  views.delete_student,   name="delete_student"),
    path("attendance/",                  views.attendance_page,  name="attendance_page"),
    path("attendance/scan/",             views.mark_attendance,  name="mark_attendance"),
    path('attendance/export/', views.export_attendance, name='export_attendance'),
]