from operator import add
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from build_nodes import get_de_rag_pipeline, get_eu_rag_pipeline, get_us_rag_pipeline, get_news_rag_pipeline, get_web_rag_pipeline
from datetime import datetime
from textwrap import dedent
from typing import Literal
from langgraph.graph import START, END, StateGraph

class ResearchAgentState(TypedDict):
    question: str
    answers: Annotated[list[str], add]
    final_answer: str
    tools: list[str]

def de_rag_node(state: ResearchAgentState) -> ResearchAgentState:
    query = state["question"]
    docs = get_de_rag_pipeline().invoke({"query": query})
    return {"answers": [docs["generation"]]}

def eu_rag_node(state: ResearchAgentState) -> ResearchAgentState:
    query = state["question"]
    docs = get_eu_rag_pipeline().invoke({"query": query})
    return {"answers": [docs["generation"]]}

def us_rag_node(state: ResearchAgentState) -> ResearchAgentState:
    query = state["question"]
    docs = get_us_rag_pipeline().invoke({"query": query})
    return {"answers": [docs["generation"]]}

def news_rag_node(state: ResearchAgentState) -> ResearchAgentState:
    query = state["question"]
    docs = get_news_rag_pipeline().invoke({"query": query})
    return {"answers": [docs["generation"]]}

def web_rag_node(state: ResearchAgentState) -> ResearchAgentState:
    query = state["question"]
    docs = get_web_rag_pipeline().invoke({"query": query})
    return {"answers": [docs["generation"]]}

def search_tools_node(state: ResearchAgentState) -> ResearchAgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    today = datetime.today().strftime("%Y-%m-%d")

    class ToolSelector(BaseModel):
        tool: Literal["search_de_overview", "search_eu_overview", "search_us_overview", "search_news", "search_web", "llm_general"] = Field(
            description="Select the tool to use to answer the question"
        )

    class ToolSelectors(BaseModel):
        tools: list[ToolSelector] = Field(
            description="Select the tools to use to answer the question"
        )

    structured_llm_tool_selector = llm.with_structured_output(ToolSelectors)

    system_prompt = dedent(f"""
    You are an AI assistant helping with specializing in selecting the most appropriate tool to answer the question.
    Today is {today}.
    Follow these guidelines:

    - For questions specifically about Germany's economy and economic overview published, use the "search_de_overview" tool.
    - For questions specifically about European economy and economic overview published, use the "search_eu_overview" tool.
    - For questions specifically about United States economy and economic overview published, use the "search_us_overview" tool.
    - For questions about very recent economic news and events within the last 24 hours, use the "search_news" tool.
    - For any other economic information, including questions that need the most up-to-date data or information not covered in the overviews, use the "search_web" tool.
    - If a question is a general question not only about the economy, use the "llm_general" tool.
    Always choose all of the appropriate tools based on the user's question. 
    If a question is about a law but doesn't seem to be asking about specific legal provisions, include both the relevant law search tool and the search_web tool.""")

    tool_selector_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {query}")
    ])

    result = structured_llm_tool_selector.invoke(tool_selector_prompt.format(query=state["question"]))
    return {"tools": [tool.tool for tool in result.tools]}

def generate_answer_node(state: ResearchAgentState) -> ResearchAgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    system_prompt = """
    You are an expert in the economy.
    You will be given a question: {question} and a list of answers:{answers}.
    You need to generate an final answer to the question based on the answers.
    Guidelines:
        1. Use information from the given answers.
        2. If the answers are not enough to answer the question, you can reference to the web search result or generate based on your knowledge.
        3. Cite the source of information for each sentence in your answer. Use the following format:
            - For germany economy answers: "Germany Economy Overview"
            - For europe economy answers: "Europe Economy Overview"
            - For united states economy answers: "United States Economy Overview"
            - For news answers: "News Title" , "Source URL", and "Published Date"
            - For web answers: "Source Title" , "Source URL"
        4. Don't speculate or add information not in the answers.
        5. Keep answers concise and clear.
        6. Omit irrelevant information.
        7. If multiple answers provide the same information, cite all relevant answers.
        8. If information comes from multiple answers, combine them coherently while citing each answer.

    Make sure to answer more user-friendly as chatbot.
    Explain your answer in a way that is easy to understand with details.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\nAnswers: {answers}")
    ])

    result = llm.invoke(prompt.format(question=state["question"], answers="\n".join(state["answers"])))
    return {"final_answer": result.content}

def route_tools_node(state: ResearchAgentState) -> list[str]:
    tools = set(state["tools"])
    valid_tools = {"search_de_overview", "search_eu_overview", "search_us_overview", "search_news", "search_web", "llm_general"}
    if tools.issubset(valid_tools):
        return list(tools)
    
    return list(valid_tools)

def llm_general_node(state: ResearchAgentState) -> ResearchAgentState:
    llm = ChatOpenAI(model="gpt-4o-mini")
    docs = llm.invoke(state["question"])
    return {"final_answer": docs.content}

def get_economy_agent():
    builder = StateGraph(ResearchAgentState)
    builder.add_node("search_tools", search_tools_node)
    builder.add_node("search_de_overview", de_rag_node)
    builder.add_node("search_eu_overview", eu_rag_node)
    builder.add_node("search_us_overview", us_rag_node)
    builder.add_node("search_news", news_rag_node)
    builder.add_node("search_web", web_rag_node)
    builder.add_node("llm_general", llm_general_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "search_tools")
    builder.add_conditional_edges(
        "search_tools",
        route_tools_node,
        ["search_de_overview", "search_eu_overview", "search_us_overview", "search_news", "search_web", "llm_general"]
    )

    for tool in ["search_de_overview", "search_eu_overview", "search_us_overview", "search_news", "search_web", "llm_general"]:
        builder.add_edge(tool, "generate_answer")

    builder.add_edge("generate_answer", END)

    return builder.compile()