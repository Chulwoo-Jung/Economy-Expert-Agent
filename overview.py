from langchain_core.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os
from glob import glob
from typing import Literal
from langchain_core.documents import Document

class Overview:
    def __init__(self, hg_model_name:str="BAAI/bge-m3", oa_model_name:str="text-embedding-3-small"):
        self.embedding_hf = HuggingFaceEmbeddings(model_name=hg_model_name)
        self.embedding_oa = OpenAIEmbeddings(model=oa_model_name)
        self.chunker = SemanticChunker(embeddings=self.embedding_oa, chunk_size=100, chunk_overlap=20)
  

    def split_overview(self, path:str, name:str) -> list[Document]:
        split_docs = []
     
        loader = PyPDFLoader(path)
        pages = loader.load()
        print(f"loaded {name} with {len(pages)} pages")
            
        metadata = {
            "source": path,
            "name": name
        }

        chunks = self.chunker.split_documents(pages)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk.page_content, metadata=metadata))

        print(f"split {name} into {len(split_docs)} chunks")

        return split_docs

    def save_docs_to_chroma_db(self, path:str, name:str):
        split_docs = self.split_overview(path, name)

        Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_hf,
            collection_name=name,
            persist_directory="chroma_db"
        )
        print(f"save {name} to chroma_db")

    def get_chroma_db(self, collection_name:str):
        chroma_db = Chroma(
            embedding_function=self.embedding_hf,
            collection_name=collection_name,
            persist_directory="chroma_db"
        )
        return chroma_db

