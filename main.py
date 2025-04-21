from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import os
from typing import TypedDict, List, Any
from langchain_huggingface import HuggingFaceEmbeddings


# --- Setup API Keys ---
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAga6Pr8nyUcwEl_LnMCee1PXr6bhtgjo8"

class ResearchState(TypedDict):
    question: str
    documents: List[Any]
    retriever: Any
    answer: str


# --- Agent 1: Research Agent ---
tavily_tool = TavilySearchResults(k=5)

def research_node(state: dict):
    query = state["question"]
    search_results = tavily_tool.invoke({"query": query})

    # Make sure search_results are parsed correctly
    documents = []
    for entry in search_results:
        if isinstance(entry, dict):
            content = entry.get("content", "")
            url = entry.get("url", "unknown")
        else:
            content = str(entry)
            url = "unknown"
        documents.append(Document(page_content=content, metadata={"source": url}))

    return {"question": query, "documents": documents}


# --- Filter and Chunking ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def filter_node(state: dict):
    documents = state["documents"]
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    return {"question": state["question"], "retriever": retriever}


# --- Agent 2: Answer Drafter ---
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful research assistant. Use the following context to answer the question.

    Question: {question}

    Context:
    {context}

    Answer:
    """
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

def draft_answer_node(state: dict):
    retriever = state["retriever"]
    question = state["question"]
    docs = retriever.invoke(question)  # Updated for deprecation fix
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(question=question, context=context)
    answer = llm.invoke(prompt)
    return {"question": question, "answer": answer.content}


# --- LangGraph Setup ---
graph = StateGraph(ResearchState)
graph.add_node("research", RunnableLambda(research_node))
graph.add_node("filter", RunnableLambda(filter_node))
graph.add_node("draft", RunnableLambda(draft_answer_node))

graph.set_entry_point("research")
graph.add_edge("research", "filter")
graph.add_edge("filter", "draft")
graph.add_edge("draft", END)

app = graph.compile()


# --- Run the System ---
def run_research_pipeline(question: str):
    result = app.invoke({"question": question})
    return result["answer"]


# --- Example Use ---
if __name__ == "__main__":
    query = "What are the latest advancements in brain-computer interfaces?"
    final_answer = run_research_pipeline(query)
    print("\n--- Final Answer ---\n")
    print(final_answer)
