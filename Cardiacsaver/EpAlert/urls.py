from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('about/', views.about, name="about"),
    path('form/', views.form, name="form"),
    path('results/', views.results, name="results"),
    path('model/', views.model, name="model"),
]