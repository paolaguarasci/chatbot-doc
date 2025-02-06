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

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def create_doc_chat_recursive(url, config):
    """
    Creates a documentation chat system using LangChain and Groq
    """
    from langchain_community.document_loaders import RecursiveUrlLoader
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse

    print(f"Loading and indexing {url}...")
    print("This might take a few minutes for sites with many pages...")
    
    # Extract base domain to limit scraping
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Configure recursive loader
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading pages...", total=None)
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=2,  # Maximum recursion depth
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" ", strip=True),
            prevent_outside=True,  # Prevents going outside initial domain
            use_async=True,  # Uses async loading for better speed
            exclude_dirs=(  # Excludes common directories that don't contain documentation
                "/assets/", "/static/", "/js/", "/css/", 
                "/images/", "/img/", "/fonts/", "/search/"
            )
        )
        
        try:
            documents = loader.load()
            progress.stop()
            console.print(f"\n[green]✓ Loaded {len(documents)} pages from the site.[/green]")
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]✗ Error loading pages: {str(e)}[/red]")
            console.print("[yellow]Trying to load only the main page...[/yellow]")
    
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

def chat_loop_recursive(chain):
    """
    Starts an interactive chat loop
    """
    console.print(Panel.fit(
        "[cyan]Welcome to Doc Chat![/cyan]\n"
        "You can start asking questions about the documentation.\n"
        "Type [bold red]exit[/bold red] to quit.",
        title="Doc Chat",
        border_style="blue"
    ))
    
    while True:
        user_input = Prompt.ask("\n[bold green]You")
        
        if user_input.lower() == 'exit':
            console.print("\n[cyan]Goodbye! 👋[/cyan]")
            break
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing response...", total=None)
                response = chain.invoke(user_input)
            
            # Split response into sources and content
            parts = response.split("Assistant's Answer:", 1)
            if len(parts) == 2:
                sources, content = parts
                # Show sources in a panel
                console.print(Panel(
                    Markdown(sources.strip()),
                    title="[yellow]Sources used[/yellow]",
                    border_style="yellow"
                ))
                # Show response in a panel
                console.print(Panel(
                    Markdown(content.strip()),
                    title="[cyan]Response[/cyan]",
                    border_style="blue"
                ))
            else:
                console.print(Panel(
                    Markdown(response),
                    title="[cyan]Response[/cyan]",
                    border_style="blue"
                ))
                
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")