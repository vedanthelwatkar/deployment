# services.py
import os
import random
from django.core.mail import send_mail
import os 
import faiss
import ntpath
import tempfile
import openai
import numpy as np
import json
import random
from django.http import JsonResponse
from django.test import RequestFactory
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from rest_framework_simplejwt.tokens import RefreshToken

def send_otp(email):
    gen_otp = str(random.randint(100000, 999999))
    subject = "Welcome to our DocGPT Website"
    text = "Your OTP is: " + gen_otp
    from_email = "vedanthelwatkar@gmail.com"
    to_email = [str(email)]
    send_mail(subject, text, from_email, to_email)
    return gen_otp

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

    return most_similar_doc_index, most_similar_doc

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
