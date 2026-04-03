from django.urls import path
from . import views

urlpatterns = [
    path("", views.behavior_page, name="behavior"),
    path("monitor/", views.start_monitor, name="start_monitor"),
]
