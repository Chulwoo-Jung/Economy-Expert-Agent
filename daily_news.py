from os import getenv
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain_core.documents import Document

load_dotenv()

class NewsAgent:
    def __init__(self):
        self.api_key = getenv("NEWS_API_KEY")
        self.newsapi = NewsApiClient(api_key=self.api_key)

    def get_daily_news(self, symbol:str="economy") -> list[Document]:
        news =  self.newsapi.get_top_headlines(q=symbol, language="en", sources='bbc-news')
        return [ Document(
            page_content=f"{article['title']}\n\n{article.get('description', '')}", 
            metadata={"source": article["url"], "name": article['title'], "date": article['publishedAt']}) 
            for article in news["articles"] ]
