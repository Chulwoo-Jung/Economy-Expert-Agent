# Economy Analysis Agent

A comprehensive LangGraph-based agent system for analyzing economic data from multiple sources including Germany, EU, and US economic overviews, news, and web search.

## 🏗️ Architecture

The system consists of several interconnected components that work together to provide comprehensive economic analysis:

### Core Components

- **`overview.py`** - Handles PDF document processing and ChromaDB vector storage
- **`tools.py`** - Defines search tools and retrievers for different data sources
- **`build_nodes.py`** - Contains RAG pipeline implementations for each data source
- **`economy_agent.py`** - Main orchestrator that coordinates all components
- **`daily_news.py`** - News API integration for real-time economic news
- **`web_retriever.py`** - Web search capabilities for additional information

### Data Sources

- **Germany Economy Overview** (`de_economy` collection)
- **EU Economy Overview** (`eu_economy` collection) 
- **US Economy Overview** (`us_economy` collection)
- **Real-time News** (via NewsAPI)
- **Web Search** (via Tavily API)

## 🔄 How It Works

1. **Document Processing**: PDFs are loaded, chunked, and stored in ChromaDB with embeddings
2. **Query Analysis**: User queries are analyzed to determine relevant data sources
3. **Parallel Retrieval**: Multiple RAG pipelines run simultaneously to gather information
4. **Information Extraction**: Each pipeline extracts and evaluates relevant information
5. **Answer Synthesis**: All results are combined into a comprehensive final answer

## 🚀 Quick Start

### Prerequisites

```bash
pip install langgraph langchain-openai langchain-chroma langchain-huggingface
pip install newsapi-python sentence-transformers chromadb
```

### Environment Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key
TAVILY_API_KEY=your_tavily_key
```

### Usage

```python
from economy_agent import get_economy_agent

# Initialize the agent
agent = get_economy_agent()

# Query the system
result = agent.invoke({"question": "What is the GDP of the United States?"})
print(result['final_answer'])
```

## 🧪 Testing with Gradio

The system includes a Gradio interface for interactive testing:

```python
# Run the Gradio app
python gradio_app.py
```

The Gradio interface allows you to:
- Test different economic queries
- See which tools are selected for each query
- View the final synthesized answers
- Monitor the agent's decision-making process

## 📊 Example Queries

- "What is the current inflation rate in Germany?"
- "How is the EU economy performing?"
- "What are the latest economic news?"
- "Compare the economic growth of US and EU"

## 🔧 Configuration

### Model Settings

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-optimized)
- **LLM**: `gpt-4o-mini` for cost efficiency
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`


## 🛠️ Development

### Customizing Retrieval

Modify the retrieval parameters in `tools.py`:

```python
# Adjust retrieval settings
base_retriever=overview.get_chroma_db("de_economy").as_retriever(
    search_kwargs={"k": 5}  # Number of documents to retrieve
)
```

## 📁 Project Structure

```
├── economy/                 # PDF documents
│   ├── de_overview.pdf
│   ├── eu_overview.pdf
│   └── us_overview.pdf
├── chroma_db/              # Vector database
├── overview.py             # Document processing
├── tools.py               # Search tools & retrievers
├── build_nodes.py         # RAG pipelines
├── economy_agent.py       # Main orchestrator
├── daily_news.py          # News integration
├── web_retriever.py       # Web search
└── test.ipynb            # Testing notebook
```

### Performance Tips

- Use the Gradio interface for interactive testing
- Monitor memory usage with large documents
- Consider adjusting chunk sizes for different document types
