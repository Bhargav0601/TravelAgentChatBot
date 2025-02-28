import os
import logging
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('./.env')

def get_openai_api_key():
    logger.info("Attempting to retrieve OpenAI API key")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in .env file or environment variables.")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI API key retrieved successfully")
    return api_key

def get_markdown_content(url):
    logger.info(f"Fetching markdown content from URL: {url}")
    scrape_url = f"https://r.jina.ai/{url}"
    response = requests.get(scrape_url)
    logger.info(f"Markdown content fetched. Status code: {response.status_code}")
    return response.text

def extract_restaurants(markdown_content, openai_api_key):
    logger.info("Extracting restaurant information from markdown content")
    prompt_template = PromptTemplate(
        input_variables=["chunk"],
        template="""
        Extract restaurant information from the following markdown content:
        {chunk}
        
        Format each restaurant as:
        "Restaurant Name - Restaurant - Address - Arrondissement - Cuisine Type | Link"
        
        If no valid restaurant information is found, return "No restaurants found in this chunk."
        """
    )

    llm = ChatOpenAI(temperature=0, api_key=openai_api_key, model="gpt-3.5-turbo-16k")
    chain = LLMChain(llm=llm, prompt=prompt_template)

    logger.info("Splitting markdown content into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=200)
    chunks = text_splitter.split_text(markdown_content)
    logger.info(f"Content split into {len(chunks)} chunks")

    results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_result = chain.invoke({"chunk": chunk})
        results.append(chunk_result['text'])

    logger.info("Restaurant extraction completed")
    return "\n".join(results)

def parse_result(result):
    logger.info("Parsing extracted restaurant information")
    lines = result.strip().split("\n")
    data = []
    for line in lines:
        if "No restaurants found in this chunk." in line:
            continue
        parts = line.split("|")
        if len(parts) == 2:
            info = parts[0].strip()
            link = parts[1].strip()
            data.append({"info": info, "link": link})
    logger.info(f"Parsed {len(data)} restaurant entries")
    return pd.DataFrame(data)

def extract_restaurant_details(url, openai_api_key):
    logger.info(f"Extracting details for restaurant URL: {url}")
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="""
        Analyze the following markdown content and extract structured information about a restaurant. Please format the information into categories and provide clear and specific data for each category:

        Restaurant Details:
        Name: (Provide the full name of the restaurant)
        Location: (Specify the complete address)
        Cuisine Type: (Describe the type of food served)
        Average Price: (Mention the average cost for a meal)
        Special Offers: (List any current special offers or discounts)

        Reviews and Ratings:
        Overall Rating: (Give the overall rating out of 10)
        Category Ratings: (Provide ratings out of 10 for Food, Service, Ambiance)
        Review Highlights: (Summarize the most significant comments from reviews)
        Sample Reviews: (List at least three individual reviews, briefly summarized)

        Menu Items and Prices:
        List of menu items with their prices: (Detail several key menu items along with their prices)
        Link to the complete menu: (Provide a URL to the full menu if available)

        Additional Information:
        Reservation Details: (Describe how to make a reservation, if available)
        Gift Cards Availability: (State if gift cards are available and how they can be purchased)
        Special Features: (Mention any notable features like a terrace, romantic setting, or availability of English-speaking staff)
        Frequently Asked Questions (FAQs): (List and answer common questions related to the restaurant)

        Ensure that the data is organized and succinctly summarized. If certain information is not available from the content provided, clearly state 'Data not available' under the relevant category.

        Content:
        {content}

        Summarize the extracted information, focusing on clarity and utility.
        If the data is not found do not include don't include data.
        """
    )

    llm = ChatOpenAI(temperature=0, api_key=openai_api_key, model="gpt-3.5-turbo-16k")
    chain = LLMChain(llm=llm, prompt=prompt_template)

    logger.info("Fetching content of the restaurant page")
    content = get_markdown_content(url)

    logger.info("Splitting content into manageable chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)

    logger.info(f"Content split into {len(chunks)} chunks")
    details = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_result = chain.invoke({"content": chunk})
        details.append(chunk_result['text'])

    logger.info("Combining extracted details from all chunks")
    combined_details = "\n".join(details)
    return combined_details

def main():
    logger.info("Starting the main process")
    openai_api_key = get_openai_api_key()
    url_to_scrape = "https://www.thefork.fr/restaurants/paris-c415144/5e-arrondissement-a531"

    logger.info("Fetching markdown content...")
    markdown_content = get_markdown_content(url_to_scrape)

    logger.info("Extracting restaurant information...")
    extracted_info = extract_restaurants(markdown_content, openai_api_key)

    logger.info("Parsing results...")
    df = parse_result(extracted_info)

    logger.info("Extracting additional details for each restaurant...")
    descriptions = []
    for i, link in enumerate(df['link']):
        logger.info(f"Processing restaurant {i+1}/{len(df)}")
        url = link.split('(')[1].split(')')[0]
        description = extract_restaurant_details(url, openai_api_key)
        descriptions.append(description)

    df['description'] = descriptions
    df['chunk'] = df['info'] + '|' + df['link'] + '|' + df['description']
    df.drop(columns=['info', 'link', 'description'], inplace=True)

    logger.info("Extraction process completed")
    logger.info(f"Total restaurants extracted: {len(df)}")

    logger.info("Saving results to CSV")
    df.to_csv('scrapped_with_details.csv', index=False)
    logger.info("Results saved to 'scrapped_with_details.csv'")

if __name__ == "__main__":
    main()
