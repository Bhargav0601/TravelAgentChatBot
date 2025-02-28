import os
import logging
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found. Please ensure your .env file is configured correctly.")
        raise ValueError("API key is missing")
    return api_key

def extract_link_from_markdown(markdown):
    # Extract the URL from markdown link format
    start = markdown.find('(') + 1
    end = markdown.find(')', start)
    if start > 0 and end > start:
        return markdown[start:end]
    else:
        logger.error(f"Failed to extract link from markdown: {markdown}")
        return None

def get_markdown_content(url, retries=3, delay=5):
    while retries > 0:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"Failed to fetch content from {url}, status code {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        retries -= 1
    logger.error(f"Failed to fetch content from {url} after several retries.")
    return None

def extract_restaurant_details(content, openai_api_key):
    if content is None:
        return "Details could not be fetched due to connection issues."

    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="""
        Analyze the following markdown content and extract structured information about a restaurant:
        Content: {content}
        Extract detailed information about the restaurant, including name, location, cuisine type, average price, menu items, and reviews.
        """
    )

    llm = ChatOpenAI(temperature=0, api_key=openai_api_key, model="gpt-3.5-turbo-16k")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({"content": content})
    return result['text']

def main():
    openai_api_key = get_openai_api_key()
    df = pd.read_csv('C:/Users/Bhargav/TravelAgentChatBot/chat-agent/scrapped.csv')
    details = []

    for index, row in df.iterrows():
        logger.info(f"Fetching details for {row['info']}")
        link = extract_link_from_markdown(row['link'])
        if link:
            content = get_markdown_content(link)
            restaurant_details = extract_restaurant_details(content, openai_api_key)
            details.append(restaurant_details)
        else:
            details.append("Link extraction failed.")

    df['Details'] = details
    df.to_csv('detailed_restaurants.csv', index=False)
    logger.info("Detailed restaurant data has been saved to 'detailed_restaurants.csv'.")

if __name__ == "__main__":
    main()
