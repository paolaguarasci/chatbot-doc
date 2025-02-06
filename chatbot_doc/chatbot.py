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
    Crea un sistema di chat per la documentazione specificata usando LangChain e Groq
    """
    print(f"Caricamento e indicizzazione di {url}...")
    
    # Carica il contenuto del sito web
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # Dividi il testo in chunks usando la configurazione
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    splits = text_splitter.split_documents(documents)
    
    # Usa OpenAI per gli embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Crea il retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Inizializza il modello Groq con la configurazione
    llm = ChatGroq(
        temperature=config["temperature"],
        model_name=config["model_name"],
    )
    
    # Crea il template per il prompt
    template = """Sei un assistente esperto che aiuta a rispondere a domande sulla documentazione.
    Usa il contesto fornito per rispondere alla domanda.
    Se non puoi rispondere usando il contesto, dillo onestamente.
    
    Contesto: {context}
    
    Domanda: {question}
    
    Risposta assistente:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Crea la chain
    chain = (
        {
            "context": retriever | (lambda docs: "\n\nFonti utilizzate:\n" + "\n".join([
                f"\nURL: {doc.metadata.get('source', 'N/A')}\nTesto utilizzato:\n{doc.page_content}\n"
                for doc in docs
            ])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Sistema pronto!")
    return chain

def chat_loop(chain):
    """
    Avvia un loop di chat interattivo
    """
    print("\nBenvenuto! Puoi iniziare a fare domande sulla documentazione.")
    print("Scrivi 'exit' per uscire.\n")
    
    while True:
        user_input = input("Tu: ")
        
        if user_input.lower() == 'exit':
            print("Arrivederci!")
            break
        
        try:
            response = chain.invoke(user_input)
            print("\nBot:", response, "\n")
        except Exception as e:
            print(f"\nErrore: {str(e)}\n")

