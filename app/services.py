import os
import random
from django.core.mail import send_mail
import os 
import faiss
import ntpath
import openai
import numpy as np
import random
from django.core.mail import send_mail
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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

import requests
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}
def get_vectorstore(text_chunks):
    response = requests.post(api_url, headers=headers, json={"inputs": text_chunks, "options":{"wait_for_model":True}})
    response_json = response.json()

    embeddings = response_json

    # Create an index
    embedding_size = len(embeddings[0])
    index = faiss.IndexFlatIP(embedding_size)

    # Convert embeddings to NumPy array and add them to the index
    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    # Save the index to a file
    faiss.write_index(index, "vectorstore.index")

    return index

    # model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    # embedding_size = 768

    # # Create an index
    # index = faiss.IndexFlatIP(embedding_size)

    # # Embed and index the text chunks
    # embeddings = []
    # for chunk in text_chunks:
    #     embedding = model.encode([chunk])[0]
    #     embeddings.append(embedding)

    # vectors = np.array(embeddings, dtype=np.float32)
    # index.add(vectors)

    # # Save the index to a file
    # faiss.write_index(index, "vectorstore.index")

    # return index


def load_vectorstore():
    # Load the index from the file
    index = faiss.read_index("vectorstore.index")
    return index


def get_similar_docs(query, text_chunks, index, k=1):
    # Encode the query using Hugging Face API
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = "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"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(api_url, headers=headers, json={"inputs": [query], "options": {"wait_for_model": True}})
    response_json = response.json()
    print(response.json)
    query_embedding = response_json[0]

    # Perform a similarity search using Faiss
    distances, similar_doc_indices = index.search(np.array([query_embedding], dtype=np.float32), k)

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
        return ". ".join(sentences[:5]).strip()
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
