"""
URL configuration for Quizogen project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from quiz.views import front,check_ans_form,que_gen_form

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',front,name='front'),
    path('que_gen/',que_gen_form,name='que_gen_form'),
    path('check/',check_ans_form,name='check_ans_form'),
    
]
