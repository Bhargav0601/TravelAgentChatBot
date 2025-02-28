from flask import Flask, request, jsonify
import os
from urllib.request import urlretrieve
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import openai

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "sk-proj-Z3cwbQurIaQHmCAEZjMtT3BlbkFJ9YMSsMuDJ2m0ElUkKGss"

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
    
    # Generate a response using OpenAI API
    prompt = f"""
    You are an expert on U.S. Census data. Based on the following documents, please provide a detailed answer to the question: {query}
    
    Documents:
    {combined_docs}
    
    Answer:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on U.S. Census data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    generated_answer = response.choices[0].message['content'].strip()
    
    return jsonify({"response": generated_answer})

if __name__ == '__main__':
    app.run(debug=True)


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
    
    # Generate a response using OpenAI API
    prompt = f"""
    You are an expert on U.S. Census data. Based on the following documents, please provide a detailed answer to the question: {query}
    
    Documents:
    {combined_docs}
    
    Answer:
    """
    
    response = openai.Completion.create(
        model="davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    generated_answer = response.choices[0].text.strip()
    
    return jsonify({"response": generated_answer})

if __name__ == '__main__':
    app.run(debug=True)
