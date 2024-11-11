import requests
from bs4 import BeautifulSoup
import pandas as pd
import openai
import os

# File paths
external_data_path = 'data/external/sports_articles_data.csv'
augmented_data_path = 'data/external/augmented_sports_articles_data.csv'

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
openai.api_key = OPENAI_API_KEY

# Step 1: Web Scraping to Collect External Data
def get_sports_data_from_web(url):
    print(f"Collecting data from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract information such as article titles, summaries, and publication dates
        articles = []
        for article in soup.find_all('article'):
            title = article.find('h2').get_text(strip=True) if article.find('h2') else ""
            summary = article.find('p').get_text(strip=True) if article.find('p') else ""
            date = article.find('time')['datetime'] if article.find('time') else "Unknown"
            
            if title:
                articles.append({'title': title, 'summary': summary, 'publication_date': date})
        
        print(f"Collected {len(articles)} articles.")
        return articles
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return []

# Example URL (replace with a valid URL for data collection)
sports_url = 'https://www.sportsnewswebsite.com/category/basketball'
external_articles = get_sports_data_from_web(sports_url)

# Save the articles to a CSV file for further analysis
if external_articles:
    external_articles_df = pd.DataFrame(external_articles)
    external_articles_df.to_csv(external_data_path, index=False)
    print(f"External articles data saved to {external_data_path}.")

# Step 2: Using OpenAI for Augmenting Data
def generate_summary_with_openai(article_list, max_tokens=100):
    summaries = []
    for article in article_list:
        prompt = (
            f"Summarize the following sports article in a concise manner and provide key takeaways:\n"
            f"Title: {article['title']}\n"
            f"Summary: {article['summary']}\n"
        )
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            generated_summary = response['choices'][0]['text'].strip()
            summaries.append({
                'title': article['title'],
                'original_summary': article['summary'],
                'generated_summary': generated_summary,
                'publication_date': article['publication_date']
            })
        except Exception as e:
            print(f"Error generating summary for article '{article['title']}': {e}")
    return summaries

# Use OpenAI to generate summaries of the collected articles
if external_articles:
    augmented_data = generate_summary_with_openai(external_articles)
    if augmented_data:
        augmented_data_df = pd.DataFrame(augmented_data)
        augmented_data_df.to_csv(augmented_data_path, index=False)
        print(f"Augmented sports articles data saved to {augmented_data_path}.")
    else:
        print("No summaries generated.")

print("RAG Data Retrieval Completed.")