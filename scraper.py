import requests
from bs4 import BeautifulSoup

# Define the URL of the website you want to scrape
url = "https://www.yelp.fr/paris"

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Raise an exception if there's an error with the request

# Parse the content of the request with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Define a list to store the scraped restaurant data
restaurants = []

# Find all the restaurant entries on the page
# This is just an example; you need to adjust the selectors based on the website's structure
for restaurant_entry in soup.find_all('div', class_='restaurant-entry'):
    name = restaurant_entry.find('h2', class_='restaurant-name').get_text(strip=True)
    address = restaurant_entry.find('p', class_='restaurant-address').get_text(strip=True)
    rating = restaurant_entry.find('span', class_='restaurant-rating').get_text(strip=True)
    
    # Store the extracted data in a dictionary and append to the list
    restaurant_info = {
        'name': name,
        'address': address,
        'rating': rating
    }
    restaurants.append(restaurant_info)

# Print the scraped data
for restaurant in restaurants:
    print(f"Name: {restaurant['name']}")
    print(f"Address: {restaurant['address']}")
    print(f"Rating: {restaurant['rating']}")
    print("-" * 40)