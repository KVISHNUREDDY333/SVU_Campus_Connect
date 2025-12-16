import json
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURATION ---
DATA_FILE = "data.json"
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "svu_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_vector_store():
    """
    Loads data from a JSON file, generates embeddings, and stores them in a ChromaDB vector store.
    """
    # 1. Load the data from data.json
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        faqs = data.get("faqs", [])
        if not faqs:
            print("No FAQs found in the data file.")
            return
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading or parsing {DATA_FILE}: {e}")
        return

    print(f"Loaded {len(faqs)} FAQs from {DATA_FILE}.")

    # 2. Initialize the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
        return

    # 3. Create documents and metadatas for ChromaDB
    documents = []
    metadatas = []
    ids = []
    for faq in faqs:
        # Combine question and answer for a richer embedding
        combined_text = f"Question: {faq['question']} Answer: {faq['answer']}"
        documents.append(combined_text)
        metadatas.append({
            "question": faq.get("question", ""),
            "answer": faq.get("answer", ""),
            "category": faq.get("category", "Uncategorized"),
            "source": faq.get("source", "Unknown"),
        })
        ids.append(faq.get("id", None))

    # Filter out any entries that failed to get an ID
    valid_indices = [i for i, faq_id in enumerate(ids) if faq_id is not None]
    if len(valid_indices) != len(faqs):
        print(f"Warning: {len(faqs) - len(valid_indices)} FAQs were skipped due to missing 'id'.")
    
    documents = [documents[i] for i in valid_indices]
    metadatas = [metadatas[i] for i in valid_indices]
    ids = [ids[i] for i in valid_indices]

    if not documents:
        print("No valid documents to process after filtering.")
        return

    # 4. Generate embeddings
    print("Generating embeddings for all documents... (This may take a while)")
    try:
        embeddings = model.encode(documents, show_progress_bar=True)
    except Exception as e:
        print(f"Error encoding documents: {e}")
        return

    # 5. Initialize ChromaDB and create the collection
    print(f"Initializing ChromaDB in directory: {DB_DIRECTORY}...")
    try:
        client = chromadb.PersistentClient(path=DB_DIRECTORY)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        print("Please ensure 'chromadb' is installed (`pip install chromadb`)")
        return

    # 6. Add the data to the collection
    # ChromaDB's add method can handle upserts if the IDs already exist.
    print(f"Adding {len(documents)} documents to the '{COLLECTION_NAME}' collection...")
    try:
        collection.add(
            embeddings=embeddings.tolist(), # Convert numpy array to list
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        print(f"Error adding data to Chroma collection: {e}")
        return

    print("\n-----------------------------------------")
    print("Vector store creation complete!")
    print(f"Total documents processed: {len(documents)}")
    print(f"Database stored in: '{DB_DIRECTORY}'")
    print(f"Collection name: '{COLLECTION_NAME}'")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    create_vector_store()
    print("To use the chatbot, run the FastAPI server: uvicorn main:app --reload")
