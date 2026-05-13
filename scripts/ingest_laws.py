"""
Ingests LFT and CPEUM Markdown files into ChromaDB collection 'leyes_mexicanas'.

Usage (from project root, with venv active):
    python scripts/ingest_laws.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "leyes_mexicanas"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 100

# H1 → titulo, H2 → capitulo, H3 → articulo (preserva jerarquía en metadatos)
HEADERS_TO_SPLIT = [
    ("#", "titulo"),
    ("##", "capitulo"),
    ("###", "articulo"),
]


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def load_md_files() -> list:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        str(PROCESSED_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        sys.exit(f"ERROR: No se encontraron archivos .md en {PROCESSED_DIR}")

    print(f"  {len(docs)} archivo(s) cargado(s):")
    for d in docs:
        print(f"    - {Path(d.metadata['source']).name}")
    return docs


# ---------------------------------------------------------------------------
# 2. DIVISIÓN (CHUNKING)
# ---------------------------------------------------------------------------

def split_into_chunks(docs: list) -> list:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,  # conserva el encabezado "### Artículo X" en el texto
    )

    all_chunks = []
    for doc in docs:
        law_name = Path(doc.metadata["source"]).stem  # "LFT" o "CPEUM"
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Garantiza que el nombre de la ley esté siempre en los metadatos
            chunk.metadata["fuente"] = law_name
        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# 3. VECTORIZACIÓN E INGESTA
# ---------------------------------------------------------------------------

def build_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"  Modelo: {EMBED_MODEL}")
    print("  (Primera ejecución descarga ~120 MB — solo ocurre una vez)")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_to_chroma(chunks: list) -> int:
    import chromadb

    print(f"\n  Conectando a ChromaDB en {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()  # falla temprano si el servicio no está disponible
    print("  Conexión OK")

    embed_fn = build_embedding_model()

    # Re-indexación limpia: elimina la colección si ya existe
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Colección '{COLLECTION_NAME}' previa eliminada (re-indexación limpia).")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "descripcion": "LFT y CPEUM para RAG de cláusulas abusivas",
        },
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    total_batches = -(-len(texts) // BATCH_SIZE)  # división techo

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]
        batch_ids = [f"doc_{i + j}" for j in range(len(batch_texts))]

        embeddings = embed_fn.embed_documents(batch_texts)

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=batch_ids,
        )

        batch_num = i // BATCH_SIZE + 1
        print(f"  Lote {batch_num}/{total_batches}: {len(batch_texts)} chunks subidos")

    return collection.count()


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main():
    separator = "=" * 52
    print(separator)
    print("  Ingesta de Leyes Mexicanas → ChromaDB")
    print(separator)

    print("\n[1/3] Cargando archivos .md...")
    docs = load_md_files()

    print("\n[2/3] Dividiendo en chunks por artículo...")
    chunks = split_into_chunks(docs)

    article_chunks = [c for c in chunks if c.metadata.get("articulo")]
    print(f"  Total chunks generados      : {len(chunks)}")
    print(f"  Chunks de artículo (### )   : {len(article_chunks)}")
    print(f"  Chunks de preámbulo/sección : {len(chunks) - len(article_chunks)}")

    per_law = {}
    for c in chunks:
        name = c.metadata.get("fuente", "desconocido")
        per_law[name] = per_law.get(name, 0) + 1
    for law, count in sorted(per_law.items()):
        print(f"    {law}: {count} chunks")

    print("\n[3/3] Vectorizando e indexando en ChromaDB...")
    total = ingest_to_chroma(chunks)

    print(f"\n{separator}")
    print("  INDEXACION COMPLETA")
    print(f"  Artículos totales en ChromaDB : {total}")
    print(separator)


if __name__ == "__main__":
    main()
