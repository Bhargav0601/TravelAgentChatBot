from flask import Flask, request, jsonify
import os
from urllib.request import urlretrieve
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download and load documents
def download_and_load_documents():
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

    loader = PyPDFDirectoryLoader("./us_census/")
    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    docs_after_split = text_splitter.split_documents(docs_before_split)
    return docs_after_split

# Initialize embeddings and vectorstore
def initialize_vectorstore(docs):
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(docs, huggingface_embeddings)

# Download documents and initialize vectorstore
docs = download_and_load_documents()
vectorstore = initialize_vectorstore(docs)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Perform similarity search
        relevant_documents = vectorstore.similarity_search(query, k=3)
        results = [doc.page_content for doc in relevant_documents]
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
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)