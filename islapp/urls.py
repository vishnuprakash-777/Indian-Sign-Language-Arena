from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_login, name='user_login'),
    path('register/', views.user_register, name='user_register'),
    path('home/', views.home, name='home'),
    path('logout/', views.user_logout, name='user_logout'),
    path('logout2/', views.admin_logout, name='admin_logout'),
    path('videos/', views.video_gallery, name='video_gallery'),
    path('upload/', views.upload_video, name='upload_video'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('predict/', views.predict_action, name='predict_action'),
    path('adminlogin/', views.admin_login, name='admin_login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('login/add/', views.add_login_detail, name='add_login_detail'),
    path('login/edit/<int:user_id>/', views.edit_login_detail, name='edit_login_detail'),
    path('login/delete/<int:user_id>/', views.delete_login_detail, name='delete_login_detail'),

    # GestureVideo
    path('gesture/add/', views.add_gesture_video, name='add_gesture_video'),
    path('gesture/edit/<int:serial_no>/', views.edit_gesture_video, name='edit_gesture_video'),
    path('gesture/delete/<int:serial_no>/', views.delete_gesture_video, name='delete_gesture_video'),

]
