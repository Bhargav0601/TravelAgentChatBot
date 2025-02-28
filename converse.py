from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv('./.env')

app = Flask(__name__)

# Initialize ChatOpenAI
chat = ChatOpenAI(
    model='gpt-3.5-turbo'
)

# Initialize Qdrant and embeddings
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load dataset and documents
dataset = pd.read_csv('C:/Users/Bhargav/TravelAgentChatBot/chat-agent/scrapped_with_details.csv') 
docs = dataset[['chunk']]
loader = DataFrameLoader(docs, page_content_column="chunk")
documents = loader.load()

qdrant = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,
    url=url,
    collection_name="chatbot",
    api_key=api_key
)

location = "Paris"
# Initialize messages with initial context
initial_context ="""
        You are a friendly chatbot travel agent for the destination: "{location}".
        If destination is "none", try to understand the user's destination of interest.
        If the user asks anything not related to travel and tourism, try to steer the conversation back to travel and tourism by saying sorry as an AI language model.  
    
        Your job is to incrementally discover and remember the user's conversation goals and topics of interest and provide the information they are looking for.

        The user is likely looking for information about the place, its history and culture, things to do, events, places to stay and places to eat and drink that align with their interests. 
        Hidden gems are likely to be of interest too. The user may or may not know anything about the place already.

        Using information from the user, try to make recommendations as you proceed.

        Ask good, open-ended questions to discover their interests and goals.
        Try to keep the conversation relevant to the following topics if any related to travel and toursim, activities, events, places to stay, places to eat and drink, hidden gems, and local culture.
        If the user asks about anything unrelated to the data provided about the destination, and restaurants realated to this location, try to steer the conversation back to the data provided about the destination, and restaurants realated to this location.

        In the response, if the data contains a [Link], provide the link to the user.
        Do not answer questions outside this domain, if the user asks a question outside this domain, ask them to ask a question related to restaurants in "{location}" and their interests.
        Continue the conversation following these instructions:

        Each output is 75 characters or less.
        Write a large variety of outputs, and potential followup questions.
        Output is relevant to the last few messages of the conversation.
        Output has easily identifiable keyword searches.
        End each output with a blank line followed by one relevant followup open-ended question to elicit more information.

        If there are any special considerations or restrictions such as elderly adults or young children and what would make the trip the best it could be. Use that information as the conversation evolves.
        If the user asks for a map of a place or area, provide a Google Maps URL
        If it seems like an itinerary would be helpful, ask the user if you can help them create an itinerary.

        Do not include “sorry as an AI language model” in your output

        If you learn about the ages of people, try to make age-appropriate suggestions.
        strictly answer the query using information from the source knowledge and related to restaurants in "{locaiton}".
        Do not answer questions ouytside this domain, if the user asks a question outside this domain, ask them to ask a question related to restaurants in "{location}" and their interests.

        If user asks about related services in "{location}" try to guide them the most to the best options available. If not guide them to the second best option available.
        """
messages = [
    SystemMessage(content=initial_context)
]

def custom_prompt(query: str):
    results = qdrant.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augment_prompt = f"""Using the contexts below strictly answer the query using information from the source knowledge below:

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augment_prompt

@app.route('/chat', methods=['POST'])
def chat_api():
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    # Ensure to copy the initial context
    current_messages = messages[:]
    prompt = HumanMessage(content=user_input)
    current_messages.append(prompt)
    
    res = chat.invoke(current_messages)
    messages.append(res)
    
    return jsonify({"response": res.content})

@app.route('/custom_chat', methods=['POST'])
def custom_chat_api():
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    augmented_prompt = custom_prompt(user_input)
    current_messages = messages[:]
    prompt = HumanMessage(content=augmented_prompt)
    current_messages.append(prompt)

    res = chat.invoke(current_messages)
    messages.append(res)

    return jsonify({"response": res.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
