from flask import Flask, request, jsonify
import os
from urllib.request import urlretrieve
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import random

app = Flask(__name__)

# Download and load documents
os.makedirs("us_census", exist_ok=True)
files = [
    "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
]

for url in files:
    file_path = os.path.join("us_census", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# Load and process documents
loader = PyPDFDirectoryLoader("./us_census/")
docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
docs_after_split = text_splitter.split_documents(docs_before_split)

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

# Initialize the language model
model_name = "distilgpt2"  # Lighter model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Perform similarity search
    relevant_documents = vectorstore.similarity_search(query, k=3)
    results = [doc.page_content for doc in relevant_documents]
    
    # Combine the retrieved document contents
    combined_docs = "\n\n".join(results)
    
    # Generate a response using the language model
    prompt = f"""
    You are an expert on U.S. Census data. Based on the following documents, please provide a detailed and friendly answer to the question: {query}
    
    Documents:
    {combined_docs}
    
    Answer:
    """
    response = nlp(prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']
    
    # Generate a conversational response
    answer_start = response.find("Answer:") + len("Answer:")
    generated_answer = response[answer_start:].strip()
    
    return jsonify({"response": f"{generated_answer}"})

if __name__ == '__main__':
    app.run(debug=True)
