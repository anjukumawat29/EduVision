from django.urls import path, include

urlpatterns = [
    path("", include("attendance.urls")),
    path("behavior/", include("behavior.urls")),
]