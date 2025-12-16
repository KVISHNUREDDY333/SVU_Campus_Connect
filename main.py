import os
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Vector Database and Embedding Model Config
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "svu_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Must match the one in create_vector_store.py
LLM_MODEL = "gemini-1.5-flash"

# --- APPLICATION INITIALIZATION ---

app = FastAPI(title="SVU CampusConnect AI")

# Global objects for models and DB client
embedding_model = None
chroma_collection = None

@app.on_event("startup")
def startup_event():
    """
    Load models and connect to the vector database on application startup.
    """
    global embedding_model, chroma_collection
    
    print("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"FATAL: Could not load SentenceTransformer model. Error: {e}")
        # In a real app, you might want to exit if the model can't be loaded.
        return

    print("Connecting to vector database...")
    try:
        if not os.path.exists(DB_DIRECTORY):
            print(f"FATAL: ChromaDB directory not found at '{DB_DIRECTORY}'.")
            print("Please run 'python create_vector_store.py' first to create the database.")
            return
            
        client = chromadb.PersistentClient(path=DB_DIRECTORY)
        chroma_collection = client.get_collection(name=COLLECTION_NAME)
        print("Successfully connected to ChromaDB and loaded collection.")
    except Exception as e:
        print(f"FATAL: Could not connect to ChromaDB. Error: {e}")
        print("Please ensure the database exists and the collection name is correct.")

# --- MIDDLEWARE ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- STATIC FILES AND FRONTEND ---

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main index.html file for the chatbot UI.
    """
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# --- API MODELS ---

class ChatRequest(BaseModel):
    message: str
    history: list = [] # Optional chat history for more context

class ChatResponse(BaseModel):
    response: str

# --- CORE RAG LOGIC ---

def find_relevant_context(query_embedding):
    """
    Queries the ChromaDB collection to find the most relevant documents.
    """
    if chroma_collection is None:
        return None
        
    results = chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5  # Retrieve top 5 most relevant documents
    )
    
    # Extract the 'documents' content from the first list of results
    return results['documents'][0] if results['documents'] else []

# --- API ENDPOINTS ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles the main chat logic using the RAG pattern.
    """
    user_query = request.message

    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    if embedding_model is None or chroma_collection is None:
        raise HTTPException(status_code=503, detail="Server is not ready. Models or DB not loaded.")

    # 1. Generate embedding for the user's query
    query_embedding = embedding_model.encode(user_query)

    # 2. Retrieve relevant context from the vector store
    context_chunks = find_relevant_context(query_embedding)
    
    if not context_chunks:
        context_str = "No relevant information found in the knowledge base."
    else:
        context_str = "\n\n".join(context_chunks)

    # 3. Construct the prompt for the LLM
    system_prompt = f"""
    You are 'CampusConnect AI', a friendly and helpful assistant for Sri Venkateswara University.
    Your task is to answer the user's question based *only* on the provided context.
    The context contains relevant Questions and Answers from the university's knowledge base.
    
    - Answer clearly and concisely.
    - If the context does not contain the answer, you MUST state: 
      "I'm sorry, I could not find a specific answer to your question in the knowledge base. For more details, you may visit the official SVU website at https://svuniversity.edu.in/."
    - Do not invent information or answer based on your general knowledge.
    - Be friendly and professional.

    Provided Context:
    ---
    {context_str}
    ---
    """
    
    # For now, we are not using the history, but it's available for future context-aware conversations.
    prompt = f"{system_prompt}\nUser Question: {user_query}"

    # 4. Generate the response using Gemini
    try:
        llm = genai.GenerativeModel(LLM_MODEL)
        response = llm.generate_content(prompt)
        return ChatResponse(response=response.text)
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a response from the AI model.")

# This allows running the app with `python main.py`
if __name__ == "__main__":
    import uvicorn
    # Note: Uvicorn is used for development. For production, a more robust server like Gunicorn is recommended.
    print("Starting FastAPI server...")
    print("Run 'python create_vector_store.py' if you haven't already.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
