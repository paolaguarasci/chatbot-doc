from chatbot_doc.chatbot import create_doc_chat, chat_loop
from chatbot_doc.recursive import create_doc_chat_recursive, chat_loop_recursive
from chatbot_doc.setup import setup_environment

# from rich import print
# from rich.theme import Theme
# from rich.style import Style
# from rich.text import Text

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.prompt import Prompt

console = Console()

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
    
    # chain = create_doc_chat(url, config)
    # chat_loop(chain)

    chain = create_doc_chat_recursive(url, config)
    chat_loop_recursive(chain)

if __name__ == "__main__":
    main()



