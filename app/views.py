from django.views.decorators.csrf import csrf_exempt
from .controllers import signup, login, change, otp, reset, delete_vectorstore, index, bot, internet
from rest_framework.views import APIView
from rest_framework.response import Response

class HelloWorld(APIView):
    def get(self, request):
        return Response({"message": "Hello, world!"})
    
@csrf_exempt
def signup_view(request):
    return signup(request)

@csrf_exempt
def login_view(request):
    return login(request)

@csrf_exempt
def change_view(request):
    return change(request)

@csrf_exempt
def otp_view(request):
    return otp(request)

@csrf_exempt
def reset_view(request):
    return reset(request)

@csrf_exempt
def delete_vectorstore_view(request):
    return delete_vectorstore(request)

@csrf_exempt
def index_view(request):
    return index(request)

@csrf_exempt
def bot_view(request):
    return bot(request)

@csrf_exempt
def internet_view(request):
    return internet(request)
