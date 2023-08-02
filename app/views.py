from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import os,faiss,ntpath,tempfile,tempfile,openai,numpy as np,json,random
from django.core.mail import send_mail
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['auth']
collection = db['signupdata']


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
            print("email = " + str(email))
            password = login_data["password"]
            print("password = " + str(password))
            subject = "Welcome to the DocGPT Website"
            text = "Here you can upload your own files and search for the query and get the most relevent answer from your provided pdfs"
            from_email = "vedanthelwatkar@gmail.com"
            to_email = [str(email)]
            send_mail(subject, text, from_email, to_email)
            
            user = collection.find_one({"email": email, "password": password})
            if user:
                message = "Login successful."
                print("logged in")
            else:
                message = "Invalid email or password."
            return JsonResponse({'message': message})
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


load_dotenv(find_dotenv())
openai_api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = openai_api_key

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    embedding_size = 768

    # Create an index
    index = faiss.IndexFlatIP(embedding_size)

    # Embed and index the text chunks
    embeddings = []
    for chunk in text_chunks:
        embedding = model.encode([chunk])[0]
        embeddings.append(embedding)

    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    # Save the index to a file
    faiss.write_index(index, "vectorstore.index")

    return index


def load_vectorstore():
    # Load the index from the file
    index = faiss.read_index("vectorstore.index")
    return index


def get_similar_docs(query, text_chunks, index, k=1):
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    query_embedding = model.encode([query])[0]
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Perform a similarity search to retrieve similar document indices and distances
    distances, similar_doc_indices = index.search(query_embedding.reshape(1, -1), k)

    # Check if the similar_doc_indices array is empty
    if len(similar_doc_indices) == 0:
        most_similar_doc_index = None
        most_similar_doc = None
    else:
        # Convert the similar_doc_indices array to a list and access the first element
        similar_doc_indices = similar_doc_indices.flatten().tolist()
        most_similar_doc_index = similar_doc_indices[0]
        most_similar_doc = text_chunks[most_similar_doc_index]

    return distances, most_similar_doc_index, most_similar_doc



def extract_answer_from_text(most_similar_doc):
    sentences = most_similar_doc.split(".")
    print("Sentences:", sentences)  # Print the sentences variable

    if len(sentences) >= 2:
        return ". ".join(sentences[:2]).strip()
    elif sentences:
        return sentences[0].strip()
    else:
        return None

def get_reference(most_similar_doc_index, text_chunks, pdf_filenames):
    pdf_file_index = most_similar_doc_index // len(text_chunks[0])
    pdf_filename = pdf_filenames[pdf_file_index]
    pdf_name = ntpath.basename(pdf_filename)
    reference = os.path.splitext(pdf_name)[0]
    return reference

@csrf_exempt
def delete_vectorstore(request):
    if request.method == "POST":
        if os.path.isfile("vectorstore.index"):
            os.remove("vectorstore.index")
            return JsonResponse({"message": "Vector Store Deleted"})
        else:
            print("vectorstore.index not found.")
            return JsonResponse({"message": "Vector Store Not Found"})
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)



@csrf_exempt
def index(request):
    text_chunks = None
    vectorstore = None
    if request.method == "POST" and "pdfFiles" in request.FILES:
        # if os.path.isfile("vectorstore.index"):
        #     return JsonResponse({"message":"delete file first"})
        print('you are here')
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
        request.session['text_chunks'] = text_chunks
        request.session['pdf_filenames'] = pdf_filenames
        request.session.flush()
        print("Length of text_chunks:", len(text_chunks))
        
        if os.path.isfile("vectorstore.index"):
            vectorstore = load_vectorstore()
        else:
            vectorstore = get_vectorstore(text_chunks)

        for pdf_file in pdf_docs:
            os.remove(pdf_file)

        return JsonResponse({"message":"Vector Store Created"})
        
    if request.method == "POST":
        text_chunks = request.session.get('text_chunks', [])
        print("Length of text_chunks in session:", len(text_chunks))
        print("Contents of text_chunks in session:", text_chunks)
        pdf_filenames = request.session.get('pdf_filenames', [])
        vectorstore = load_vectorstore()
        print("agaya bhai")
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        print(query)
        

        if not query:
            return JsonResponse({"message": "Invalid query"})

        most_similar_doc_index, most_similar_doc = get_similar_docs(query, text_chunks, vectorstore)
        if most_similar_doc_index is not None:
            most_similar_doc = text_chunks[most_similar_doc_index]
            answer = extract_answer_from_text(most_similar_doc)
            reference = get_reference(most_similar_doc_index, text_chunks, pdf_filenames) + ".pdf"

            return JsonResponse({
                "query": query,
                "answer": answer,
                "reference": reference,
                "sentences": most_similar_doc
            })
    return JsonResponse({"message": "nothing happened"})

def generate_response(user_input):
    prompt = "User: {}\nChatGPT:".format(user_input)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=50,
    )
    bot_response = response.choices[0].text.strip()
    return bot_response


def generate_bot_response(user_input):
    prompt = "User: {}\nChatGPT:".format(user_input)
    response = openai.Completion.create(
        model="davinci",
        prompt=prompt,
        temperature=0.6,
        max_tokens=50,
    )
    bot_response = response.choices[0].text.strip()
    return bot_response


@csrf_exempt
def internet(request):
    if request.method == "POST":
        data = json.loads(request.body)
        message = data.get("query", "").strip()
        result = generate_response(message)
        return JsonResponse({"result": result})
    return JsonResponse({"result": "nothing happened"})