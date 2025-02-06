from typing import List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

def create_doc_chat(url: str, config: dict):
    """
    Creates a documentation chat system using LangChain and Groq
    """
    print(f"Loading and indexing {url}...")
    # Load website content
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split text into chunks using configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    splits = text_splitter.split_documents(documents)

    # Use OpenAI for embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Initialize Groq model with configuration
    llm = ChatGroq(
        temperature=config["temperature"],
        model_name=config["model_name"],
    )

    # Create prompt template
    template = """You are an expert assistant who helps answer questions about documentation.
    Use the provided context to answer the question.
    If you cannot answer using the context, say so honestly.
    Context: {context}
    Question: {question}
    Assistant's Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {
            "context": retriever | (lambda docs: "\n\nSources used:\n" + "\n".join([
                f"\nURL: {doc.metadata.get('source', 'N/A')}\nText used:\n{doc.page_content}\n"
                for doc in docs
            ])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("System ready!")
    return chain

def chat_loop(chain):
    """
    Starts an interactive chat loop
    """
    print("\nWelcome! You can start asking questions about the documentation.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = chain.invoke(user_input)
            print("\nBot:", response, "\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")