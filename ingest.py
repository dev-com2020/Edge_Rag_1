# ingest.py
import chromadb
from openai import OpenAI

# Klient Ollama do generowania embeddingów
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key='lmstudio',
 )

# Użyj modelu embeddingowego dostępnego w Ollama
EMBEDDING_MODEL = "s3dev-ai/text-embedding-nomic-embed-text-v1.5"

# Połącz się z ChromaDB (plik zostanie stworzony w folderze vector_store)
db = chromadb.PersistentClient(path="./vector_store")

# Stwórz lub załaduj kolekcję
collection = db.get_or_create_collection(name="rag_wiedza")

# Wczytaj i podziel dokument
with open("data/wiedza.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Prosty chunking po liniach
chunks = [line for line in text.split('\n') if line.strip()]

# Generuj embeddingi i dodaj do kolekcji
for i, chunk in enumerate(chunks):
    response = client.embeddings.create(
        input=chunk,
        model=EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("Baza wiedzy została pomyślnie załadowana do ChromaDB.")