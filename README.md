# 🍽️ Chat Agent: Your AI Restaurant Recommendation Companion

Introducing our state-of-the-art Chat Agent, a revolutionary plug-and-play system designed to enhance your restaurant recommendation experience. This system is meticulously crafted to cater to both businesses and individuals seeking accurate, personalized dining suggestions within a predefined area.

## 🚀 Key Features:

1. **🕸️ Dynamic Data Scraping:**
   - Continuously scrapes up-to-date restaurant information from various online sources within a specified geographic area.
   - Collects comprehensive data including restaurant names, cuisines, ratings, reviews, locations, and more.

2. **📊 Advanced Data Analysis:**
   - Analyzes the scraped restaurant data to extract meaningful insights and trends.
   - Employs sophisticated algorithms to assess restaurant quality and popularity based on user reviews and ratings.

3. **🔍 Vector Database Integration:**
   - Stores analyzed restaurant data in a high-performance vector database.
   - Uses vector similarity search to match user queries with the most relevant restaurant recommendations.
   - Ensures rapid and precise search results by leveraging the power of vector embeddings.

4. **🧠 Contextual Enrichment:**
   - Continuously enriches the vector database with new and updated information.
   - Enhances the context of existing data to improve recommendation accuracy and relevance.

5. **💬 Chatbot Interface:**
   - Provides an intuitive, user-friendly chatbot interface for seamless interaction.
   - Easily integratable into agency and client websites, offering a personalized recommendation service.
   - Responds to user queries with tailored restaurant suggestions based on the enriched data.

## 👥 Target Audience:

### 🏢 Businesses:
- Restaurant owners and managers seeking to promote their establishments.
- Hospitality and tourism agencies aiming to provide value-added services to their customers.
- Marketing agencies looking to enhance their clients' online presence with personalized recommendations.

### 👤 Individuals:
- Food enthusiasts searching for new dining experiences.
- Tourists and visitors seeking local dining recommendations.
- Residents looking for the best restaurants in their area.

## 💎 Benefits:

### 📈 For Businesses:
- Increases visibility and customer engagement through accurate and personalized recommendations.
- Enhances the user experience on websites, driving traffic and potential revenue growth.
- Offers a competitive edge by utilizing cutting-edge technology in customer service.

### 🎉 For Individuals:
- Saves time by providing quick and reliable restaurant suggestions tailored to personal preferences.
- Ensures up-to-date and comprehensive information for making informed dining choices.
- Offers a convenient and enjoyable way to discover new culinary experiences.

Our Chat Agent with RAG technology sets a new standard in restaurant recommendation systems, combining the latest in data scraping, analysis, and vector similarity search to deliver unparalleled results. Whether you're a business looking to attract more customers or an individual in search of the perfect dining spot, our system is your ultimate solution.

# 🛠️ Installation and Running Instructions

## Setting Up the Environment

1. Create a new conda environment:
conda create -n rag_chatbot python=3.9

2. Activate the conda environment:
conda activate rag_chatbot

3. Install the required dependencies:
pip install -r requirements.txt

4. Clone the repository or download the project files:
git clone git@github.com:Your-Name/TravelAgentChatBot.git
cd your-repo-name

## 🚀 Running the Application

1. Run the scraper to collect restaurant data (this is a one-time or periodic operation):
python scraper.py
CopyNote: The scraper is designed to be run sequentially and may take some time depending on the amount of data to be collected.

2. Start the backend conversation module:
python converse.py

⚠️ Important: Ensure that the backend (converse.py) is running before launching the chat interface.

4. Launch the Streamlit chat interface:
streamlit run chat-interface.py

After running the Streamlit command, your default web browser should open automatically, displaying the chat interface. If it doesn't, you can manually open a browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## 🔍 Vector Database Visualization

You can view and analyze the vector embeddings stored in our Qdrant database. This gives you insight into how restaurants are represented in the vector space and how similarity is computed.

To access the Qdrant dashboard:

1. Open your web browser
2. Navigate to: https://f1a30225-fd67-4bb6-9fac-994236a0cadd.us-east4-0.gcp.cloud.qdrant.io:6333/dashboard#/collections/chatbot
3. Here you can explore the `chatbot` collection, which contains the vector embeddings for our restaurant data.

## 📈 Performance Monitoring with Grafana

We use Grafana to monitor the performance and health of our chat agent system. You can access detailed metrics and dashboards to track usage, response times, and other key performance indicators.

To access the Grafana dashboard:

1. Ensure that the Grafana service is running
2. Open your web browser
3. Navigate to: http://localhost:3000
4. Log in with your credentials (if required)
5. Explore the various dashboards to gain insights into system performance

By leveraging these analytics tools, you can:
- Understand how restaurants are clustered in the vector space
- Monitor system performance and identify bottlenecks
- Track usage patterns and popular queries
- Make data-driven decisions to improve the recommendation engine

Note: Ensure that you're in the project directory when running these commands, and that your conda environment is activated.

## 🛡️ Input Validation with Giskard

We use Giskard to validate inputs to our recommendation system. This helps ensure that:

- User queries are properly formatted and sanitized
- Input data meets our quality standards
- Potential edge cases are handled gracefully

Giskard helps us maintain the integrity of our system by catching and addressing potential issues before they affect the recommendation process.

## 📊 Model Monitoring with LLM Guard

To continuously improve our RAG chain, we employ LLM Guard to monitor specific metrics of our language model's performance. This includes:

- Response quality and relevance
- Bias detection and mitigation
- Safety checks for generated content
- Performance metrics such as latency and token usage

LLM Guard allows us to:
- Identify areas for improvement in our model
- Ensure the ethical use of our AI system
- Optimize resource usage and response times
- Maintain high standards of recommendation quality

By integrating these tools, we create a robust feedback loop that continuously enhances our RAG chain, resulting in better restaurant recommendations and an improved user experience.


Happy restaurant hunting! 🍽️🎉
