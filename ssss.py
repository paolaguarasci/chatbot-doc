import os
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

def setup_environment():
    """
    Configura l'ambiente e verifica le variabili necessarie
    """
    # Cerca il file .env nella directory corrente e nelle directory parent
    dotenv_path = find_dotenv()
    if not dotenv_path:
        # Se non trova un .env, ne crea uno nuovo nella directory corrente
        default_env_content = """# Configurazione per Doc Chat
# Inserisci le tue API keys
GROQ_API_KEY=
OPENAI_API_KEY=

# Configurazione opzionale
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MODEL_NAME=mixtral-8x7b-32768
TEMPERATURE=0.1

# Configurazione scraping
MAX_DEPTH=2  # Profondità massima di ricorsione
PREVENT_OUTSIDE=true  # Impedisce di uscire dal dominio principale
"""
        with open(".env", "w") as f:
            f.write(default_env_content)
        print("File .env creato. Per favore, inserisci le tue API keys e riavvia il programma.")
        exit(1)
    
    # Carica le variabili d'ambiente
    load_dotenv(dotenv_path)
    
    # Verifica le variabili necessarie
    required_vars = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Errore: Le seguenti variabili d'ambiente sono mancanti: {', '.join(missing_vars)}")
        print(f"Per favore, aggiungile al file .env in: {dotenv_path}")
        exit(1)
    
    # Carica le configurazioni opzionali con valori di default
    config = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
        "model_name": os.getenv("MODEL_NAME", "mixtral-8x7b-32768"),
        "temperature": float(os.getenv("TEMPERATURE", 0.1)),
        "max_depth": int(os.getenv("MAX_DEPTH", 2)),
        "prevent_outside": os.getenv("PREVENT_OUTSIDE", "true").lower() == "true"
    }
    
    return config

def create_doc_chat(url: str, config: dict):
    """
    Crea un sistema di chat per la documentazione specificata usando LangChain e Groq
    """
    from langchain_community.document_loaders import RecursiveUrlLoader
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse

    print(f"Caricamento e indicizzazione di {url}...")
    print("Questo potrebbe richiedere alcuni minuti per siti con molte pagine...")
    
    # Estrai il dominio base per limitare lo scraping
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Configura il loader ricorsivo
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Caricamento pagine...", total=None)
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=2,  # Profondità massima di ricorsione
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" ", strip=True),
            prevent_outside=True,  # Impedisce di uscire dal dominio iniziale
            use_async=True,  # Usa loading asincrono per maggiore velocità
            exclude_dirs=(  # Esclude directory comuni che non contengono documentazione
                "/assets/", "/static/", "/js/", "/css/", 
                "/images/", "/img/", "/fonts/", "/search/"
            )
        )
        
        try:
            documents = loader.load()
            progress.stop()
            console.print(f"\n[green]✓ Caricate {len(documents)} pagine dal sito.[/green]")
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]✗ Errore durante il caricamento delle pagine: {str(e)}[/red]")
            console.print("[yellow]Provo a caricare solo la pagina principale...[/yellow]")
    
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
    console.print(Panel.fit(
        "[cyan]Benvenuto nel Doc Chat![/cyan]\n"
        "Puoi iniziare a fare domande sulla documentazione.\n"
        "Scrivi [bold red]exit[/bold red] per uscire.",
        title="Doc Chat",
        border_style="blue"
    ))
    
    while True:
        user_input = Prompt.ask("\n[bold green]Tu")
        
        if user_input.lower() == 'exit':
            console.print("\n[cyan]Arrivederci! 👋[/cyan]")
            break
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Elaborazione risposta...", total=None)
                response = chain.invoke(user_input)
            
            # Divide la risposta in fonti e contenuto
            parts = response.split("Risposta assistente:", 1)
            if len(parts) == 2:
                sources, content = parts
                # Mostra le fonti in un panel
                console.print(Panel(
                    Markdown(sources.strip()),
                    title="[yellow]Fonti utilizzate[/yellow]",
                    border_style="yellow"
                ))
                # Mostra la risposta in un panel
                console.print(Panel(
                    Markdown(content.strip()),
                    title="[cyan]Risposta[/cyan]",
                    border_style="blue"
                ))
            else:
                console.print(Panel(
                    Markdown(response),
                    title="[cyan]Risposta[/cyan]",
                    border_style="blue"
                ))
                
        except Exception as e:
            console.print(f"\n[red]Errore: {str(e)}[/red]")

def main():
    # Setup dell'ambiente e caricamento configurazione
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Caricamento configurazione...", total=None)
        config = setup_environment()
        progress.stop()
    
    # Richiedi l'URL e avvia il sistema
    url = Prompt.ask("\n[bold cyan]Inserisci l'URL della documentazione")
    chain = create_doc_chat(url, config)
    chat_loop(chain)

if __name__ == "__main__":
    main()