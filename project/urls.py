from django.urls import path
from app import views
from app.views import HelloWorld

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('change/', views.change_view, name='change'),
    path('otp/', views.otp_view, name='otp'),
    path('reset/', views.reset_view, name='reset'),
    path('delete_vectorstore/', views.delete_vectorstore_view, name='delete_vectorstore'),
    path('', views.index_view, name='index'),
    path('bot/', views.bot_view, name='bot'),
    path('internet/', views.internet_view, name='internet'),
    path('hello/', HelloWorld.as_view(), name='hello-world'),
]
