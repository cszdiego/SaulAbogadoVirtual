"""Verifica la conexión con ChromaDB y crea la colección de leyes."""
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)

heartbeat = client.heartbeat()
print(f"ChromaDB OK — heartbeat: {heartbeat}")

collection = client.get_or_create_collection(
    name="leyes_mexicanas",
    metadata={"descripcion": "LFT y CPEUM para RAG de cláusulas abusivas"},
)
print(f"Colección '{collection.name}' lista. Documentos: {collection.count()}")
