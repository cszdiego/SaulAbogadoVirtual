"""
Baseline evaluation: RAG + LLM para detectar cláusulas abusivas en contratos mexicanos.

Detecta automáticamente el backend de inferencia disponible:
  - GPU con CUDA  →  LLaMA-3-8B cargado con bitsandbytes 4-bit (transformers)
  - CPU (sin GPU) →  modelo vía Ollama (llama3.2:3b por defecto)

Prerequisitos comunes:
  - Docker corriendo con ChromaDB (docker compose up -d)
  - Colección 'leyes_mexicanas' indexada (python scripts/ingest_laws.py)

Prerequisito GPU:
  - GPU NVIDIA con CUDA >= 8 GB VRAM + drivers CUDA instalados

Prerequisito CPU (sin GPU):
  - Ollama instalado: https://ollama.com/download/windows
  - Modelo descargado: ollama pull llama3.2:3b

Ejecución (desde la raíz del proyecto, con venv activo):
    python scripts/eval_baseline.py
"""
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

_contracts_dir = BASE_DIR / "data" / "processed" / "Contracts"
CONTRACT_PATH = next(
    (p for p in _contracts_dir.glob("*malas_practicas.md") if p.is_file()),
    None,
)

REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_PATH = REPORTS_DIR / "baseline_analysis.md"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "leyes_mexicanas"

# GPU backend
GPU_MODEL_ID = "unsloth/llama-3-8b-instruct-bnb-4bit"

# CPU backend (Ollama)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TOP_K_ARTICLES = 5
MAX_NEW_TOKENS = 1024
MAX_CONTRACT_CHARS = 6000

SYSTEM_PROMPT = (
    "Eres un experto legal mexicano. Tu tarea es encontrar irregularidades en el "
    "contrato adjunto basándote exclusivamente en los artículos de la LFT proporcionados. "
    "Si no encuentras una violación directa, indícalo."
)


# ---------------------------------------------------------------------------
# 1. DETECCIÓN DE BACKEND
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """
    Determina el backend de inferencia disponible.
    Retorna: 'cuda' | 'ollama'
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        pass

    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL.split(":")[0] in m for m in models):
                print(f"  Ollama detectado — modelo '{OLLAMA_MODEL}' disponible")
                return "ollama"
            else:
                # Ollama está corriendo pero el modelo no está descargado
                print(f"  Ollama disponible, descargando '{OLLAMA_MODEL}'...")
                _ollama_pull(OLLAMA_MODEL)
                return "ollama"
    except requests.exceptions.ConnectionError:
        pass

    # Ningún backend disponible
    print("\n[ERROR] No se encontró ningún backend de inferencia.\n")
    print("  Opción A — Con GPU NVIDIA:")
    print("    Instala los drivers CUDA: https://developer.nvidia.com/cuda-downloads")
    print()
    print("  Opción B — Sin GPU (recomendado para tu máquina):")
    print("    1. Descarga Ollama para Windows: https://ollama.com/download/windows")
    print("    2. Instala y luego ejecuta en una terminal:")
    print(f"         ollama pull {OLLAMA_MODEL}")
    print("    3. Vuelve a ejecutar este script.")
    sys.exit(1)


def _ollama_pull(model: str) -> None:
    """Descarga un modelo en Ollama mostrando progreso básico."""
    with requests.post(
        f"{OLLAMA_HOST}/api/pull",
        json={"name": model},
        stream=True,
        timeout=600,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                import json
                data = json.loads(line)
                status = data.get("status", "")
                if "pulling" in status or "success" in status:
                    print(f"    {status}")


# ---------------------------------------------------------------------------
# 2. VERIFICACIONES PREVIAS
# ---------------------------------------------------------------------------

def check_prerequisites(backend: str) -> None:
    errors = []

    if CONTRACT_PATH is None:
        errors.append(
            f"No se encontró ningún archivo '*malas_practicas.md' en {_contracts_dir}"
        )

    try:
        import chromadb
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        col = client.get_collection(COLLECTION_NAME)
        if col.count() == 0:
            errors.append(
                f"La colección '{COLLECTION_NAME}' está vacía. "
                "Ejecuta primero: python scripts/ingest_laws.py"
            )
    except Exception as exc:
        errors.append(
            f"ChromaDB no disponible en {CHROMA_HOST}:{CHROMA_PORT}. "
            f"¿Está Docker corriendo? (docker compose up -d)\n  Detalle: {exc}"
        )

    if errors:
        print("\n[ERROR] No se puede continuar:\n")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 3. RAG — Recuperación de artículos relevantes
# ---------------------------------------------------------------------------

def retrieve_relevant_articles(contract_text: str) -> list[dict]:
    import chromadb
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"  Cargando embeddings ({embed_model_name})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    query_vector = embeddings.embed_query(contract_text[:2000])

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_ARTICLES,
        include=["documents", "metadatas"],
    )

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ---------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL PROMPT
# ---------------------------------------------------------------------------

def build_messages(contract_text: str, articles: list[dict]) -> tuple[str, str]:
    """
    Retorna (system_prompt, user_message) válidos tanto para Ollama (chat)
    como para el template de LLaMA-3 Instruct (GPU).
    """
    context_blocks = []
    for i, art in enumerate(articles, 1):
        fuente = art["metadata"].get("fuente", "LEY")
        articulo = art["metadata"].get("articulo", "")
        header = f"[Artículo {i} — {fuente}] {articulo}".strip()
        context_blocks.append(f"{header}\n{art['text'].strip()}")

    context_str = "\n\n---\n\n".join(context_blocks)

    contract_excerpt = contract_text[:MAX_CONTRACT_CHARS]
    if len(contract_text) > MAX_CONTRACT_CHARS:
        contract_excerpt += "\n\n[... contrato truncado por longitud ...]"

    user_message = (
        f"### ARTÍCULOS RELEVANTES DE LA LFT RECUPERADOS POR RAG:\n\n"
        f"{context_str}\n\n"
        "---\n\n"
        f"### CONTRATO A ANALIZAR:\n\n"
        f"{contract_excerpt}\n\n"
        "---\n\n"
        "Analiza cláusula por cláusula. Para cada irregularidad indica: "
        "(1) la cláusula del contrato, (2) el artículo de la LFT que se vulnera, "
        "y (3) una breve explicación."
    )

    return SYSTEM_PROMPT, user_message


# ---------------------------------------------------------------------------
# 5A. INFERENCIA — Backend Ollama (CPU)
# ---------------------------------------------------------------------------

def run_ollama_inference(system_prompt: str, user_message: str) -> str:
    print(f"  Modelo: {OLLAMA_MODEL} vía Ollama (CPU)")
    print("  Esto puede tardar 2-5 minutos en CPU — es normal.")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0,           # determinista para baseline
            "num_predict": MAX_NEW_TOKENS,
            "num_ctx": 8192,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        sys.exit("[ERROR] Ollama tardó demasiado. Prueba con un modelo más pequeño: OLLAMA_MODEL=llama3.2:1b")
    except requests.exceptions.RequestException as exc:
        sys.exit(f"[ERROR] Fallo al comunicarse con Ollama: {exc}")


# ---------------------------------------------------------------------------
# 5B. INFERENCIA — Backend GPU (CUDA + bitsandbytes)
# ---------------------------------------------------------------------------

def run_gpu_inference(system_prompt: str, user_message: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Modelo: {GPU_MODEL_ID}")
    print("  (Primera ejecución descarga ~4.5 GB — solo ocurre una vez)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GPU_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Template de chat LLaMA-3 Instruct
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Tokens de entrada: {input_len} | Máx. salida: {MAX_NEW_TOKENS}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 6. GUARDAR REPORTE
# ---------------------------------------------------------------------------

def save_report(
    backend: str,
    contract_path: Path,
    articles: list[dict],
    analysis: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_label = OLLAMA_MODEL if backend == "ollama" else GPU_MODEL_ID

    articles_section = ""
    for i, art in enumerate(articles, 1):
        meta = art["metadata"]
        fuente = meta.get("fuente", "LEY")
        articulo = meta.get("articulo", "(sin número)")
        articles_section += f"**{i}. [{fuente}]** {articulo}\n\n"
        articles_section += f"> {art['text'][:300].strip()}...\n\n"

    report = textwrap.dedent(f"""\
        # Análisis Baseline — Detección de Cláusulas Abusivas

        **Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        **Modelo:** `{model_label}` (backend: {backend.upper()})
        **Contrato analizado:** `{contract_path.name}`
        **Colección RAG:** `{COLLECTION_NAME}` ({TOP_K_ARTICLES} artículos recuperados)

        ---

        ## Artículos Recuperados por RAG

        {articles_section}
        ---

        ## Análisis del Modelo (Pre Fine-Tuning)

        {analysis}

        ---

        *Este reporte sirve como línea base (baseline) antes del fine-tuning con LoRA.*
        *Comparar con el análisis post-entrenamiento para medir la mejora del modelo.*
    """)

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"  Reporte guardado en: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 56
    print(sep)
    print("  Eval Baseline — Abogado Virtual")
    print(sep)

    print("\n[0/4] Detectando backend de inferencia...")
    backend = detect_backend()
    print(f"  Backend seleccionado: {backend.upper()}")

    print("\n[1/4] Verificando prerequisitos...")
    check_prerequisites(backend)
    print("  OK")

    print(f"\n[2/4] Cargando contrato: {CONTRACT_PATH.name}")
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    print(f"  {len(contract_text):,} caracteres leídos")

    print(f"\n[3/4] Recuperando {TOP_K_ARTICLES} artículos relevantes de ChromaDB...")
    articles = retrieve_relevant_articles(contract_text)
    for i, art in enumerate(articles, 1):
        meta = art["metadata"]
        label = f"[{meta.get('fuente','?')}] {meta.get('articulo','')[:60]}"
        print(f"  {i}. {label}")

    print(f"\n[4/4] Ejecutando inferencia ({backend.upper()})...")
    system_prompt, user_message = build_messages(contract_text, articles)

    if backend == "ollama":
        analysis = run_ollama_inference(system_prompt, user_message)
    else:
        analysis = run_gpu_inference(system_prompt, user_message)

    print("\n--- ANÁLISIS GENERADO ---")
    print(analysis)
    print("-------------------------\n")

    save_report(backend, CONTRACT_PATH, articles, analysis)

    print(f"\n{sep}")
    print("  BASELINE COMPLETO")
    print(f"  Reporte: {OUTPUT_PATH}")
    print(sep)


if __name__ == "__main__":
    main()
