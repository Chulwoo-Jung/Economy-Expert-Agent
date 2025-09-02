from overview import Overview
from daily_news import NewsAgent
from glob import glob
import os
from langchain_core.documents import Document
from langchain_core.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

overview = Overview()
news_agent = NewsAgent()
llm = ChatOpenAI(model="gpt-5-nano")

# saving documents to chroma_db

paths = glob(os.path.join("economy", "*overview.pdf"))
names = [path.split("/")[-1].split(".")[0].split("_")[0] for path in paths]
print(f"save {names} to chroma_db")

for path, name in zip(paths, names):
    overview.save_docs_to_chroma_db(path, name)

# building retrievers
re_ranker = CrossEncoderReranker(cross_encoder=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"), top_n=2)

de_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("de").as_retriever(search_kwargs={"k": 5}),
)
eu_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("eu").as_retriever(search_kwargs={"k": 5}),
)
us_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("us").as_retriever(search_kwargs={"k": 5}),
)

# defining tools

@tool
def search_de_overview(query: str) -> list[Document]:
    """search information from germany economy overview that is published in August"""
    retriever = de_retriever
    return retriever.invoke(query)

@tool
def search_eu_overview(query: str) -> list[Document]:
    """search information from europe economy overview that is published in March"""
    retriever = eu_retriever
    return retriever.invoke(query)

@tool
def search_us_overview(query: str) -> list[Document]:
    """search information from united states economy overview that is published in July"""
    retriever = us_retriever
    return retriever.invoke(query)

@tool
def search_news(query: str) -> list[Document]:
    """search information from news that is published in the last 24 hours"""
    retriever = news_agent.get_daily_news("economy")
    return retriever.invoke(query)

@tool
def search_web(query: str) -> list[Document]:
    """search information that don't exixt in the database from the web"""
    web_retriever = TavilySearchAPIRetriever(k=5)
    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f"<Document href={doc.metadata['source']}/>\n{doc.page_content}\n</Document>",
                metadata={"source": "web search", "url": doc.metadata["source"]}
            )
        )
    return formatted_docs