from django.urls import path
from . import views
 
urlpatterns = [
    path("behavior/",        views.behavior_page,  name="behavior"),
    path("behavior/start/",  views.start_monitor,  name="start_monitor"),
]
 