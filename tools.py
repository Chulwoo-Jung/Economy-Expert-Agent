from overview import Overview
from daily_news import NewsAgent
from glob import glob
import os
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.tools import tool

overview = Overview()
news_agent = NewsAgent()

# saving documents to chroma_db
paths = glob(os.path.join("economy", "*overview.pdf"))
names = [path.split("/")[-1].split(".")[0].split("_")[0] for path in paths]
# ChromaDB requires collection names to be at least 3 characters
names = [f"{name}_economy" for name in names]
print(f"save {names} to chroma_db")

for path, name in zip(paths, names):
    overview.save_docs_to_chroma_db(path, name)

# building retrievers
re_ranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"), top_n=2)

de_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("de_economy").as_retriever(search_kwargs={"k": 5}),
)
eu_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("eu_economy").as_retriever(search_kwargs={"k": 5}),
)
us_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=overview.get_chroma_db("us_economy").as_retriever(search_kwargs={"k": 5}),
)

# defining tools

@tool
def search_de_overview(query: str) -> list[Document]:
    """search information from germany economy overview that is published in August"""
    de_docs = de_retriever.invoke(query)

    if(len(de_docs) > 0):
        return de_docs
    
    return [Document(page_content="Can't find relevant information.")]

@tool
def search_eu_overview(query: str) -> list[Document]:
    """search information from europe economy overview that is published in March"""
    eu_docs = eu_retriever.invoke(query)

    if(len(eu_docs) > 0):
        return eu_docs
    
    return [Document(page_content="Can't find relevant information.")]

@tool
def search_us_overview(query: str) -> list[Document]:
    """search information from united states economy overview that is published in July"""
    us_docs = us_retriever.invoke(query)

    if(len(us_docs) > 0):
        return us_docs
    
    return [Document(page_content="Can't find relevant information.")]

@tool
def search_news(query: str) -> list[Document]:
    """search information from news that is published in the last 24 hours"""
    news_docs = news_agent.get_daily_news("economy")

    if(len(news_docs) > 0):
        return news_docs
    
    return [Document(page_content="Can't find relevant information.")]

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