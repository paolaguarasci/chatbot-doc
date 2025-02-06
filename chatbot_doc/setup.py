from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from pydantic import SecretStr

def setup_environment():
    required_vars = {
        "GROQ_API_KEY": SecretStr(os.environ["GROQ_API_KEY"]),
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Errore: Le seguenti variabili d'ambiente sono mancanti: {', '.join(missing_vars)}")
        print(f"Per favore, aggiungile al file .env in: {dotenv_path}")
        exit(1)

    config = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
        "model_name": os.getenv("MODEL_NAME", "mixtral-8x7b-32768"),
        "temperature": float(os.getenv("TEMPERATURE", 0.1)),
        "max_depth": int(os.getenv("MAX_DEPTH", 2)),
        "prevent_outside": os.getenv("PREVENT_OUTSIDE", "true").lower() == "true"
    }
    
    return config