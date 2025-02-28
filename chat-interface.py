import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# Initialize chat history and welcomed flag
if "messages" not in st.session_state:
    st.session_state.messages = []
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Function to get bot response
def get_bot_response(prompt):
    url = "http://localhost:8000/custom_chat"  # Update with your actual API endpoint
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"query": prompt})
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Sorry, I couldn't process your request."

# Function to simulate typing effect and ensure markdown renders correctly
def type_effect(text):
    placeholder = st.empty()
    full_response = ""
    for word in text.split():
        full_response += word + " "
        placeholder.markdown(full_response + "▌")
        time.sleep(0.1)  # Adjust the typing speed here
    # Adding a small delay before re-rendering the final response
    time.sleep(0.5)
    placeholder.markdown(full_response)

# Display welcome message if not already welcomed
if not st.session_state.welcomed:
    location = "Paris"  # You can change this or make it dynamic
    welcome_message = f"Hello! I'm a travel agent for {location}. Ask me about restaurants in the area!"
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    st.session_state.welcomed = True

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.write(content)

# User input
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    response = get_bot_response(prompt)
    
    # Display bot response with typing effect
    with st.chat_message("assistant"):
        type_effect(response)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.title("Chat Assistant")
    st.write("This is an AI-powered chat assistant. Feel free to ask questions about travel and tourism!")
