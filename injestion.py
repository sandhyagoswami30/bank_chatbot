import json
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

#1) Load dataset
with open("data.json", "r") as file:
    data = json.load(file)

#2) Extract questions
questions = [item["question"] for item in data]

#3) Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#4) Create embeddings
question_embeddings = model.encode(questions)


#5) Function to find most similar question
def get_most_similar(query):
    query_embedding = model.encode([query])[0]

    similarities = []
    for emb in question_embeddings:
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append(sim)

    return np.argmax(similarities)


#6) LLM function
def generate_response(context, query):
    prompt = f"""
You are a banking assistant chatbot.

Use the given context to answer the question clearly and simply.

Context: {context}

Question: {query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


#7) Chatbot loop
print("Banking Chatbot with LLM (type 'exit' to stop)\n")

while True:
    user_query = input("User: ")

    if user_query.lower() == "exit":
        print("Bot: Goodbye!")
        break

    index = get_most_similar(user_query)
    context = data[index]["answer"]

    final_answer = generate_response(context, user_query)

    print("Bot:", final_answer)