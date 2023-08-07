from django.http import JsonResponse
from .services import  get_pdf_text, get_text_chunks, get_vectorstore, load_vectorstore, get_similar_docs, extract_answer_from_text, get_reference,generate_response
import os
import tempfile
import random
import json
from django.http import JsonResponse
from django.test import RequestFactory
from dotenv import load_dotenv, find_dotenv
import openai
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from django.core.mail import send_mail

load_dotenv(find_dotenv())
openai_api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = openai_api_key

connection='mongodb+srv://vedanthelwatkar:vedant@docgpt.ptgdojj.mongodb.net/docgpt'
client = MongoClient(connection)
db = client['docgpt']
collection = db['auth']
chat = db['chat']

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        try:
            signup_data = json.loads(request.body)
            first_name = signup_data["first_name"]
            last_name = signup_data["last_name"]
            email = signup_data["email"]
            password = signup_data["password"]
            user = collection.find_one({"email": email})
            if user is not None:
                return JsonResponse({'message': 'AE'})
            else:
                data_to_insert = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "password": password,
                }
                collection.insert_one(data_to_insert)          
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        return JsonResponse({'message': 'Data received and saved successfully.'})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
@csrf_exempt
def login(request):
    if request.method == 'POST':
        try:
            login_data = json.loads(request.body)
            email = login_data["email"].lower()
            password = login_data["password"]

            user = collection.find_one({"email": email,"password":password})
            if user is None:
                return JsonResponse({'error': 'Invalid email or password.'})
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                # If the user does not exist, create a new User object
                user = User(username=email, email=email)
                user.set_password(password)  # Set the user's password
            user.save()
            refresh = RefreshToken.for_user(user)
            token = str(refresh.access_token)
            print(token)
            return JsonResponse({'message': 'Login successful.', 'token': token})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
@csrf_exempt
def change(request):
    if request.method == 'POST':
        try:
            cp = json.loads(request.body)
            email = cp["email"].lower()
            print("email = " + str(email))
            password1 = cp["password"]
            user = collection.find_one({"email": email})
            if user is not None:
                collection.update_one({"email": email}, {"$set": {"password": password1}})
            else:
                return JsonResponse({'message': 'User not found.'})
            return JsonResponse({'message': 'Password changed successfully.'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
@csrf_exempt
def otp(request):
    if request.method == 'POST':
        try:
            fp = json.loads(request.body)
            print(fp)
            email = fp["email"].lower()
            gen_otp = str(random.randint(100000, 999999))
            print(gen_otp)
            subject = "Welcome to our DocGPT Website"
            text = "Your OTP is: " + gen_otp
            from_email = "vedanthelwatkar@gmail.com"
            to_email = [str(email)]
            send_mail(subject, text, from_email, to_email)
            print("OTP sent")
            return JsonResponse({"otp": gen_otp, "message": "DONE", "to_email": to_email})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

@csrf_exempt
def reset(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(data)
            email = data["email"]
            user_otp = data["otp"]
            print(user_otp)
            new_password = data["password"]
            user = collection.find_one({"email": email})
            if user:
                collection.update_one({"email": email}, {"$set": {"password": new_password}})
                print("updated")
                return JsonResponse({'message': 'Password updated successfully.'})

            return JsonResponse({'message': 'User not found.'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)   

def delete_vectorstore(request):
    if request.method == "POST":
        if os.path.isfile("vectorstore.index"):
            os.remove("vectorstore.index")
            return JsonResponse({"message": "DELETED"})
        else:
            return JsonResponse({"message": "Vector Store Not Found"})
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)


text_chunks = None
pdf_filenames = []

def index(request):
    global text_chunks
    global pdf_filenames
    if request.method == "POST":
        pdf_files = request.FILES.getlist("pdfFiles")
        pdf_docs = []

        if not pdf_files:
            return JsonResponse({"message": "no input"})

        pdf_filenames = []
        for pdf_file in pdf_files:
            pdf_filename = pdf_file.name
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(pdf_file.read())
                pdf_docs.append(temp_pdf.name)
                pdf_filenames.append(pdf_filename)

        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        print("text chunks generated length = " + str(len(text_chunks)))

        if os.path.isfile("vectorstore.index"):
            vectorstore = load_vectorstore()
        else:
            vectorstore = get_vectorstore(text_chunks)

        for pdf_file in pdf_docs:
            os.remove(pdf_file)
        return JsonResponse({"message": "Vector Store Created"})
    
    return JsonResponse({"message": "nothing happened"})

def bot(request):
    index_request = RequestFactory().post('/index/')
    index(index_request)
    
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        vectorstore = load_vectorstore()

        if not query:
            return JsonResponse({"message": "Invalid query"})

        if text_chunks is None:
            return JsonResponse({"message": "Text chunks not available"})

        most_similar_doc_index, most_similar_doc = get_similar_docs(query, text_chunks, vectorstore)
        
        if most_similar_doc_index is None:
            return JsonResponse({"message": "No similar documents found"})

        answer = extract_answer_from_text(most_similar_doc)
        print(answer)
        reference = get_reference(most_similar_doc_index, text_chunks, pdf_filenames) + ".pdf"
        print(reference)
        if answer is None:
            return JsonResponse({"message":"server restarted"})
        if answer is not None:
            data_to_insert = {
                    "query": query,
                    "answer": answer,
                    "reference": reference,
                }
            chat.insert_one(data_to_insert)

        return JsonResponse({
            "query": query,
            "answer": answer,
            "reference": reference,
            "sentences": most_similar_doc
        })
    
    return JsonResponse({"message": "Invalid request method"})

def internet(request):
    if request.method == "POST":
        data = json.loads(request.body)
        message = data.get("query", "").strip()
        result = generate_response(message)
        return JsonResponse({"result": result})
    return JsonResponse({"result": "nothing happened"})
