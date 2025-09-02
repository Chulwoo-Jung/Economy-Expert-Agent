from tools import search_de_overview, search_eu_overview, search_us_overview, search_news, search_web
from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class InformationStrip(BaseModel):
    content: str = Field(..., description="extracted information content")
    source: str = Field(..., description="information source(legal provision or URL etc.)")
    relevance_score: float = Field(..., ge=0, le=1, description="relevance score of the extracted information to the query (0-1)")
    faithfulness_score: float = Field(..., ge=0, le=1, description="faithfulness score of the extracted information to the query (0-1)")

class ExtractedInformation(BaseModel):
    strips: List[InformationStrip] = Field(..., description="extracted information strips")
    query_relevance: float = Field(..., ge=0, le=1, description="overall relevance score of the extracted information to the query (0-1)")

class RefinedQuestion(BaseModel):
    question_refined : str = Field(..., description="refined question")
    reason : str = Field(..., description="reason")

class EconomyRetrievalState(TypedDict):
    query: str      
    rewritten_query: str
    generation: str               
    documents: list[Document] 
    extracted_info: Optional[ExtractedInformation]    
    num_generations: int   

def get_de_retrieve_documents(state: EconomyRetrievalState) -> EconomyRetrievalState:
    query = state.get("rewritten_query", state["query"])
    docs = search_de_overview(query)
    return {"documents": docs}

def get_de_extract_and_evaluate_information(state: EconomyRetrievalState) -> EconomyRetrievalState:
    extracted_info = []

    for doc in state["documents"]:
        system_prompt = """
        You are an expert in extracting information from germany economy overview.
        You will be given a document{doc} and a query{query}.
        You need to extract the information from the document that is relevant to the query.
        You need to evaluate the relevance and faithfulness of the extracted information to the query.

        Output Format:
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        ...

        Finally, you need to evaluate the overall relevance score of the extracted information to the query.

        Output Format:
        - overall relevance score: [0-1]
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {doc}\nQuery: {query}")
        ])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_with_structured_output = llm.with_structured_output(ExtractedInformation)
        result = llm_with_structured_output.invoke(prompt.format(doc=doc.page_content, query=state["query"]))

        if result.query_relevance < 0.8:
            continue
        for strip in result.strips:
            if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                extracted_info.append(strip)
        
    return {"extracted_info": extracted_info, "num_generations": state.get("num_generations", 0) + 1}

def get_de_refine_question(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in germany economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to refine the question to be more relevant to the query.
    You need to give the short reason for the refined question.

    Guidelines:
    - You need to refine the question to be more relevant to the extracted information.
    - You need to refine the question to be more relevant to the query based on the relevance and faithfulness of the extracted information.
    - You need to keep the refined question as close as possible to the original question.
    - You need to keep the refined question as concise as possible.
    - You need to keep the refined question as clear as possible.

    Output Format:
    - refined question: [refined question]
    - reason: [reason]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structured_output = llm.with_structured_output(RefinedQuestion)

    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm_with_structured_output.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    return {"rewritten_query": result.question_refined}

def get_de_generate_answer(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in germany economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to generate the answer to the query.
    Guidelines:
    - You need to generate the answer to the query.
    - You need to use the extracted information to generate the answer.
    - You need to keep the answer as concise as possible.
    - You need to keep the answer as clear as possible.
    - In case of the extracted information is not enough to generate the answer, you need to generate the answer based on the query.
    - You need to cite the source of the answer.

    Output Format:
    - content: [content]
    - source: [source] (information source(legal provision or URL etc.)) (In case of the extracted information is enough to generate the answer, you need to cite the source of the extracted information. In case of the extracted information is not enough to generate the answer, you need to cite the source of the query.)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    num_generations = state.get("num_generations", 0) + 1

    return {"generation": result.content, "num_generations": num_generations}

def should_stop(state: EconomyRetrievalState):
    if state.get("num_generations", 0) > 5:
        return 'stop'
    if state.get("extracted_info") is not None and len(state.get("extracted_info", [])) > 0:
        return 'stop'
    return 'continue'

def get_de_rag_pipeline():
    builder = StateGraph(EconomyRetrievalState)

    builder.add_node("retrieve_documents", get_de_retrieve_documents)
    builder.add_node("extract_and_evaluate_information", get_de_extract_and_evaluate_information)
    builder.add_node("refine_question", get_de_refine_question)
    builder.add_node("generate_answer", get_de_generate_answer)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "extract_and_evaluate_information")
    builder.add_conditional_edges(
        "extract_and_evaluate_information",
        should_stop,
        {
            "continue": "refine_question",
            "stop": "generate_answer"
        }
    )
    builder.add_edge("refine_question", "retrieve_documents")
    builder.add_edge("generate_answer", END)

    return builder.compile()

def get_eu_retrieve_documents(state: EconomyRetrievalState) -> EconomyRetrievalState:
    query = state.get("rewritten_query", state["query"])
    docs = search_eu_overview(query)
    return {"documents": docs}

def get_eu_extract_and_evaluate_information(state: EconomyRetrievalState) -> EconomyRetrievalState:
    extracted_info = []

    for doc in state["documents"]:
        system_prompt = """
        You are an expert in extracting information from europe economy overview.
        You will be given a document{doc} and a query{query}.
        You need to extract the information from the document that is relevant to the query.
        You need to evaluate the relevance and faithfulness of the extracted information to the query.

        Output Format:
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        ...

        Finally, you need to evaluate the overall relevance score of the extracted information to the query.

        Output Format:
        - overall relevance score: [0-1]
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {doc}\nQuery: {query}")
        ])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_with_structured_output = llm.with_structured_output(ExtractedInformation)
        result = llm_with_structured_output.invoke(prompt.format(doc=doc.page_content, query=state["query"]))

        if result.query_relevance < 0.8:
            continue
        for strip in result.strips:
            if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                extracted_info.append(strip)
        
    return {"extracted_info": extracted_info, "num_generations": state.get("num_generations", 0) + 1}

def get_eu_refine_question(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in europe economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to refine the question to be more relevant to the query.
    You need to give the short reason for the refined question.

    Guidelines:
    - You need to refine the question to be more relevant to the extracted information.
    - You need to refine the question to be more relevant to the query based on the relevance and faithfulness of the extracted information.
    - You need to keep the refined question as close as possible to the original question.
    - You need to keep the refined question as concise as possible.
    - You need to keep the refined question as clear as possible.

    Output Format:
    - refined question: [refined question]
    - reason: [reason]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structured_output = llm.with_structured_output(RefinedQuestion)

    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm_with_structured_output.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    return {"rewritten_query": result.question_refined}

def get_eu_generate_answer(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in europe economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to generate the answer to the query.
    Guidelines:
    - You need to generate the answer to the query.
    - You need to use the extracted information to generate the answer.
    - You need to keep the answer as concise as possible.
    - You need to keep the answer as clear as possible.
    - In case of the extracted information is not enough to generate the answer, you need to generate the answer based on the query.
    - You need to cite the source of the answer.

    Output Format:
    - content: [content]
    - source: [source] (information source(legal provision or URL etc.)) (In case of the extracted information is enough to generate the answer, you need to cite the source of the extracted information. In case of the extracted information is not enough to generate the answer, you need to cite the source of the query.)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    num_generations = state.get("num_generations", 0) + 1

    return {"generation": result.content, "num_generations": num_generations}

def get_eu_rag_pipeline():
    builder = StateGraph(EconomyRetrievalState)

    builder.add_node("retrieve_documents", get_eu_retrieve_documents)
    builder.add_node("extract_and_evaluate_information", get_eu_extract_and_evaluate_information)
    builder.add_node("refine_question", get_eu_refine_question)
    builder.add_node("generate_answer", get_eu_generate_answer)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "extract_and_evaluate_information")
    builder.add_conditional_edges(
        "extract_and_evaluate_information",
        should_stop,
        {
            "continue": "refine_question",
            "stop": "generate_answer"
        }
    )
    builder.add_edge("refine_question", "retrieve_documents")
    builder.add_edge("generate_answer", END)

    return builder.compile()

def get_us_retrieve_documents(state: EconomyRetrievalState) -> EconomyRetrievalState:
    query = state.get("rewritten_query", state["query"])
    docs = search_us_overview(query)
    return {"documents": docs}

def get_us_extract_and_evaluate_information(state: EconomyRetrievalState) -> EconomyRetrievalState:
    extracted_info = []

    for doc in state["documents"]:
        system_prompt = """
        You are an expert in extracting information from united states economy overview.
        You will be given a document{doc} and a query{query}.
        You need to extract the information from the document that is relevant to the query.
        You need to evaluate the relevance and faithfulness of the extracted information to the query.

        Output Format:
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        ...

        Finally, you need to evaluate the overall relevance score of the extracted information to the query.

        Output Format:
        - overall relevance score: [0-1]
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {doc}\nQuery: {query}")
        ])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_with_structured_output = llm.with_structured_output(ExtractedInformation)
        result = llm_with_structured_output.invoke(prompt.format(doc=doc.page_content, query=state["query"]))

        if result.query_relevance < 0.8:
            continue
        for strip in result.strips:
            if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                extracted_info.append(strip)
        
    return {"extracted_info": extracted_info, "num_generations": state.get("num_generations", 0) + 1}

def get_us_refine_question(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in united states economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to refine the question to be more relevant to the query.
    You need to give the short reason for the refined question.

    Guidelines:
    - You need to refine the question to be more relevant to the extracted information.
    - You need to refine the question to be more relevant to the query based on the relevance and faithfulness of the extracted information.
    - You need to keep the refined question as close as possible to the original question.
    - You need to keep the refined question as concise as possible.
    - You need to keep the refined question as clear as possible.

    Output Format:
    - refined question: [refined question]
    - reason: [reason]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structured_output = llm.with_structured_output(RefinedQuestion)

    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm_with_structured_output.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    return {"rewritten_query": result.question_refined}

def get_us_generate_answer(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in united states economy.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to generate the answer to the query.
    Guidelines:
    - You need to generate the answer to the query.
    - You need to use the extracted information to generate the answer.
    - You need to keep the answer as concise as possible.
    - You need to keep the answer as clear as possible.
    - In case of the extracted information is not enough to generate the answer, you need to generate the answer based on the query.
    - You need to cite the source of the answer.

    Output Format:
    - content: [content]
    - source: [source] (information source(legal provision or URL etc.)) (In case of the extracted information is enough to generate the answer, you need to cite the source of the extracted information. In case of the extracted information is not enough to generate the answer, you need to cite the source of the query.)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    num_generations = state.get("num_generations", 0) + 1

    return {"generation": result.content, "num_generations": num_generations}

def get_us_rag_pipeline():
    builder = StateGraph(EconomyRetrievalState)

    builder.add_node("retrieve_documents", get_us_retrieve_documents)
    builder.add_node("extract_and_evaluate_information", get_us_extract_and_evaluate_information)
    builder.add_node("refine_question", get_us_refine_question)
    builder.add_node("generate_answer", get_us_generate_answer)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "extract_and_evaluate_information")
    builder.add_conditional_edges(
        "extract_and_evaluate_information",
        should_stop,
        {
            "continue": "refine_question",
            "stop": "generate_answer"
        }
    )
    builder.add_edge("refine_question", "retrieve_documents")
    builder.add_edge("generate_answer", END)

    return builder.compile()

def get_web_retrieve_documents(state: EconomyRetrievalState) -> EconomyRetrievalState:
    query = state.get("rewritten_query", state["query"])
    docs = search_web(query)
    return {"documents": docs}

def get_web_extract_and_evaluate_information(state: EconomyRetrievalState) -> EconomyRetrievalState:
    extracted_info = []

    for doc in state["documents"]:
        system_prompt = """
        You are an expert in extracting information from the web.
        You will be given a document{doc} and a query{query}.
        You need to extract the information from the document that is relevant to the query.
        You need to evaluate the relevance and faithfulness of the extracted information to the query.

        Output Format:
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        ...

        Finally, you need to evaluate the overall relevance score of the extracted information to the query.

        Output Format:
        - overall relevance score: [0-1]
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {doc}\nQuery: {query}")
        ])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_with_structured_output = llm.with_structured_output(ExtractedInformation)
        result = llm_with_structured_output.invoke(prompt.format(doc=doc.page_content, query=state["query"]))

        if result.query_relevance < 0.8:
            continue
        for strip in result.strips:
            if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                extracted_info.append(strip)
        
    return {"extracted_info": extracted_info, "num_generations": state.get("num_generations", 0) + 1}

def get_web_refine_question(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in the web.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to refine the question to be more relevant to the query.
    You need to give the short reason for the refined question.

    Guidelines:
    - You need to refine the question to be more relevant to the extracted information.
    - You need to refine the question to be more relevant to the query based on the relevance and faithfulness of the extracted information.
    - You need to keep the refined question as close as possible to the original question.
    - You need to keep the refined question as concise as possible.
    - You need to keep the refined question as clear as possible.

    Output Format:
    - refined question: [refined question]
    - reason: [reason]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structured_output = llm.with_structured_output(RefinedQuestion)

    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm_with_structured_output.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    return {"rewritten_query": result.question_refined}

def get_web_generate_answer(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in the web.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to generate the answer to the query.
    Guidelines:
    - You need to generate the answer to the query.
    - You need to use the extracted information to generate the answer.
    - You need to keep the answer as concise as possible.
    - You need to keep the answer as clear as possible.
    - In case of the extracted information is not enough to generate the answer, you need to generate the answer based on the query.
    - You need to cite the source of the answer.

    Output Format:
    - content: [content]
    - source: [source] (information source(legal provision or URL etc.)) (In case of the extracted information is enough to generate the answer, you need to cite the source of the extracted information. In case of the extracted information is not enough to generate the answer, you need to cite the source of the query.)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    num_generations = state.get("num_generations", 0) + 1

    return {"generation": result.content, "num_generations": num_generations}


def get_web_rag_pipeline():
    builder = StateGraph(EconomyRetrievalState)

    builder.add_node("retrieve_documents", get_web_retrieve_documents)
    builder.add_node("extract_and_evaluate_information", get_web_extract_and_evaluate_information)
    builder.add_node("refine_question", get_web_refine_question)
    builder.add_node("generate_answer", get_web_generate_answer)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "extract_and_evaluate_information")
    builder.add_conditional_edges(
        "extract_and_evaluate_information",
        should_stop,
        {
            "continue": "refine_question",
            "stop": "generate_answer"
        }
    )
    builder.add_edge("refine_question", "retrieve_documents")
    builder.add_edge("generate_answer", END)

    return builder.compile()    

def get_news_retrieve_documents(state: EconomyRetrievalState) -> EconomyRetrievalState:
    query = state.get("rewritten_query", state["query"])
    docs = search_news(query)
    return {"documents": docs}

def get_news_extract_and_evaluate_information(state: EconomyRetrievalState) -> EconomyRetrievalState:
    extracted_info = []

    for doc in state["documents"]:
        system_prompt = """
        You are an expert in extracting information from news.
        You will be given a document{doc} and a query{query}.
        You need to extract the information from the document that is relevant to the query.
        You need to evaluate the relevance and faithfulness of the extracted information to the query.

        Output Format:
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        - [extracted information content]
        - relevance score: [0-1]
        - faithfulness score: [0-1]
        ...

        Finally, you need to evaluate the overall relevance score of the extracted information to the query.

        Output Format:
        - overall relevance score: [0-1]
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {doc}\nQuery: {query}")
        ])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_with_structured_output = llm.with_structured_output(ExtractedInformation)
        result = llm_with_structured_output.invoke(prompt.format(doc=doc.page_content, query=state["query"]))

        if result.query_relevance < 0.8:
            continue
        for strip in result.strips:
            if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                extracted_info.append(strip)
        
    return {"extracted_info": extracted_info, "num_generations": state.get("num_generations", 0) + 1}

def get_news_refine_question(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in news.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to refine the question to be more relevant to the query.
    You need to give the short reason for the refined question.

    Guidelines:
    - You need to refine the question to be more relevant to the extracted information.
    - You need to refine the question to be more relevant to the query based on the relevance and faithfulness of the extracted information.
    - You need to keep the refined question as close as possible to the original question.
    - You need to keep the refined question as concise as possible.
    - You need to keep the refined question as clear as possible.

    Output Format:
    - refined question: [refined question]
    - reason: [reason]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structured_output = llm.with_structured_output(RefinedQuestion)

    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm_with_structured_output.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    return {"rewritten_query": result.question_refined}

def get_news_generate_answer(state: EconomyRetrievalState) -> EconomyRetrievalState:
    system_prompt = """
    You are an expert in news.
    You will be given a extracted information{extracted_info} and a query{query}.
    You need to generate the answer to the query.
    Guidelines:
    - You need to generate the answer to the query.
    - You need to use the extracted information to generate the answer.
    - You need to keep the answer as concise as possible.
    - You need to keep the answer as clear as possible.
    - In case of the extracted information is not enough to generate the answer, you need to generate the answer based on the query.
    - You need to cite the source of the answer.

    Output Format:
    - content: [content]
    - source: [source] (information source(legal provision or URL etc.)) (In case of the extracted information is enough to generate the answer, you need to cite the source of the extracted information. In case of the extracted information is not enough to generate the answer, you need to cite the source of the query.)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extracted Information: {extracted_info}\nQuery: {query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    extracted_info = "\n\n".join(strip.content for strip in state["extracted_info"])
    result = llm.invoke(prompt.format(extracted_info=extracted_info, query=state["query"]))

    num_generations = state.get("num_generations", 0) + 1

    return {"generation": result.content, "num_generations": num_generations}

def get_news_rag_pipeline():
    builder = StateGraph(EconomyRetrievalState)

    builder.add_node("retrieve_documents", get_news_retrieve_documents)
    builder.add_node("extract_and_evaluate_information", get_news_extract_and_evaluate_information)
    builder.add_node("refine_question", get_news_refine_question)
    builder.add_node("generate_answer", get_news_generate_answer)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "extract_and_evaluate_information")
    builder.add_conditional_edges(
        "extract_and_evaluate_information",
        should_stop,
        {
            "continue": "refine_question",
            "stop": "generate_answer"
        }
    )
    builder.add_edge("refine_question", "retrieve_documents")
    builder.add_edge("generate_answer", END)

    return builder.compile()