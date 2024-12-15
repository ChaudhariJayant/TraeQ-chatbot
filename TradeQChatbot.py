import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import Literal, List
import cassio
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing_extensions import TypedDict
import os 

# Set up Astra DB
astra_db_token = "AstraCS:mOEnWhhyrdPqzTuJYHSzNCws:00b6c46eb52ec9d71ba284485fb6df39f45e07ec5acf3cdba6565e5199044306"
astra_db_id = "d7a3f0c0-1490-4bb3-bf5f-c91d72de8a51"
cassio.init(token=astra_db_token, database_id=astra_db_id)

def main():
    st.title("Trading Q&A Bot")
    st.write("Ask your questions about trading strategies!")
    

    # Initialize session state for question history
    if "question_history" not in st.session_state:
        st.session_state["question_history"] = []

    # Input from user
    question = st.text_input("Enter your question:")

    # Display history
    if st.session_state["question_history"]:
        st.write("### Question History")
        for idx, q in enumerate(st.session_state["question_history"], start=1):
            st.write(f"{idx}. {q}")

    # Load Workflow
    app = load_workflow()
    
    if st.button("Submit"):
        if question.strip():
            st.session_state["question_history"].append(question)
            st.write(f"Your question: {question}")
            st.write("Processing the response...")

            # Call the workflow and get the result
            inputs = {"question": question}
            output = None  # Store the single best result

            # Stream results and stop at the first valid response
            for result in app.stream(inputs):
                if result:  # Ensure there's valid output
                    output = result
                    break

            if output:
                for key, value in output.items():
                    if key =='retrieve':
                        st.write(f"Answer from '{key}':")
                        document = value['documents'][0]
                        #st.write(type(value))
                        #page_content = document.
                        st.write(document.page_content)  # Display only the first document
                    else:
                        st.write(f"Answer from '{key}':")
            else:
                st.write("No response was generated. Try rephrasing your question.")

            st.write("Processing complete.")
        else:
            st.warning("Please enter a valid question.")

# Workflow Setup
def load_workflow():
    # Document Indexing
    urls = [
        "https://www.atfx.com/en/analysis/trading-strategies/forex-news-trading-strategy",
        "https://www.cmcmarkets.com/en/trading-guides/trading-strategies",
        "https://optimusfutures.com/blog/end-of-day-trading-strategies/",
        "https://www.captrader.com/en/blog/swing-trading-strategy/",
        "https://www.sofi.com/learn/content/day-trading-strategies/",
        "https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/trading-strategy/"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split and Embed
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=2)
    doc_splits = text_splitter.split_documents(docs_list)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(embedding=embeddings, table_name="qa_trading")
    astra_vector_store.add_documents(doc_splits)
    retriever = astra_vector_store.as_retriever()

    class GraphState(TypedDict):
        question: str
        documents: List[str]

    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "wiki_search"] = Field(
            ...,
            description="Given a user question, choose to route it to Wikipedia or a vectorstore."
        )

    # Environment Variables
    os.environ["USER_AGENT"] = "TradingQA_Bot/1.0"
    groq_api_key = "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"
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Wikipedia Setup
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Routing Prompt
    system = """
    You are an expert at routing a user question to a vectorstore or Wikipedia.
    Use the vectorstore for questions related to trading strategies, agents, or related content.
    Use Wikipedia for general knowledge questions.

    Examples:
    - "What is a day trading strategy?" -> vectorstore
    - "Who is Warren Buffett?" -> wiki_search
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router

    # Define Nodes and Workflow
    def retrieve(state):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def wiki_search(state):
        question = state["question"]
        try:
            docs = wiki.invoke({"query": question})
            if docs:
                # Inspect docs to check its structure
                st.write(docs)  # This will help us debug what the structure looks like
                # Check if docs are in the expected format and extract accordingly
                if isinstance(docs, list) and len(docs) > 0:
                    # Assume that docs is a list of dictionaries and the first item contains the text
                    document = docs[0]
                    document_text = document.get("text", "No document text found.")  # Adjust based on actual structure
                    return {"documents": [document_text], "question": question}
                else:
                    return {"documents": ["No relevant Wikipedia information found."], "question": question}
            else:
                return {"documents": ["No relevant Wikipedia information found."], "question": question}
        except Exception as e:
            st.error(f"Error in Wikipedia search: {e}")
            return {"documents": ["Error fetching Wikipedia results."], "question": question}

    def route_question(state: GraphState) -> str:
        question = state["question"]
        source = question_router.invoke({"question": question})
        return source.datasource

    # Build Workflow
    workflow = StateGraph(state_schema=GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_conditional_edges(
        START,
        route_question,
        {"vectorstore": "retrieve", "wiki_search": "wiki_search"},
    )
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)

    return workflow.compile()

if __name__ == "__main__":
    main()