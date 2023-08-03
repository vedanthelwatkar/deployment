from django.contrib import admin
from django.urls import path
from app.views import index,internet,signup,login,change,reset,otp,delete_vectorstore,bot,get_session_key

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", index, name="index"),
    path("internet/", internet, name="internet"),
    path('signup/', signup, name="signup"),
    path('login/', login, name="login"),
    path('change/', change, name="change"),
    path('reset/', reset, name="reset"),
    path('otp/', otp, name="otp"),
    path('delete_vectorstore/', delete_vectorstore, name='delete_vectorstore'),
    path('bot/',bot,name="bot"),
    path('get_session_key/', get_session_key, name='get_session_key'),
]
