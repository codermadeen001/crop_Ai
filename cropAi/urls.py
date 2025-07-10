from django.urls import path
from .views import predict
from . import views

urlpatterns = [
   path('', views.index, name='index'),
   path('api/', predict),
]



