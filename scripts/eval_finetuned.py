"""
Evaluación del modelo fine-tuneado con LoRA para detección de cláusulas abusivas.

Detecta automáticamente el backend disponible:
  - GPU con CUDA  →  modelo base + adaptadores LoRA (máxima fidelidad)
  - CPU sin GPU   →  modelo exportado a Ollama como 'abogado-virtual'
                     (requiere haber corrido scripts/export_to_gguf.py primero)

Modos de RAG:
  - Con Docker corriendo  → recupera artículos reales de ChromaDB
  - Sin Docker            → artículos de fallback hardcodeados (LFT clave)

Ejecución local:
    python scripts/eval_finetuned.py

Ejecución en Colab (GPU):
    !python eval_finetuned.py
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

ADAPTER_PATH   = str(BASE_DIR / "models" / "abogado_virtual_lora")
BASE_MODEL_ID  = "unsloth/llama-3-8b-instruct-bnb-4bit"
REPORTS_DIR    = BASE_DIR / "reports"
OUTPUT_PATH    = REPORTS_DIR / "finetuned_analysis.md"

CHROMA_HOST      = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT      = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME  = "leyes_mexicanas"

OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL     = "abogado-virtual"   # modelo creado con export_to_gguf.py

TOP_K_ARTICLES     = 3        # menos artículos = prompt más corto = más rápido
MAX_NEW_TOKENS     = 512      # respuesta más concisa
MAX_CONTRACT_CHARS = 3000     # mitad del contrato (las cláusulas clave están al inicio)

SYSTEM_PROMPT = (
    "Eres un experto legal mexicano. Tu tarea es encontrar irregularidades en el "
    "contrato adjunto basándote exclusivamente en los artículos de la LFT proporcionados. "
    "Si no encuentras una violación directa, indícalo."
)

# Artículos de fallback cuando ChromaDB no está disponible (Colab sin Docker).
# Cubren las violaciones principales del contrato de prueba.
FALLBACK_ARTICLES = [
    {
        "metadata": {"fuente": "LFT", "articulo": "Artículo 5o."},
        "text": (
            "### Artículo 5o.- Las disposiciones de esta Ley son de orden público, "
            "por lo que no producirá efecto legal, ni impedirá el goce y el ejercicio "
            "de los derechos, sea escrita o verbal, la estipulación que establezca: "
            "trabajo para menores de quince años; una jornada mayor que la permitida; "
            "salario inferior al mínimo; renuncia por parte del trabajador de cualquiera "
            "de los derechos o prerrogativas consignados en las normas de trabajo."
        ),
    },
    {
        "metadata": {"fuente": "LFT", "articulo": "Artículo 87."},
        "text": (
            "### Artículo 87.- Los trabajadores tendrán derecho a un aguinaldo anual "
            "que deberá pagarse antes del día veinte de diciembre, equivalente a quince "
            "días de salario, por lo menos. Los que no hayan cumplido el año de servicios, "
            "independientemente de que se encuentren laborando o no en la fecha de "
            "liquidación del aguinaldo, tendrán derecho a que se les pague la parte "
            "proporcional del mismo, conforme al tiempo que hubieren trabajado, "
            "cualquiera que fuere éste."
        ),
    },
    {
        "metadata": {"fuente": "LFT", "articulo": "Artículo 80."},
        "text": (
            "### Artículo 80.- Los trabajadores tendrán derecho a una prima no menor "
            "de veinticinco por ciento sobre los salarios que les correspondan durante "
            "el período de vacaciones."
        ),
    },
    {
        "metadata": {"fuente": "LFT", "articulo": "Artículo 110."},
        "text": (
            "### Artículo 110.- Los descuentos en los salarios de los trabajadores, "
            "están prohibidos salvo en los casos y con los requisitos siguientes: "
            "I. Pago de deudas contraídas con el patrón por anticipo de salarios, "
            "pagos hechos con exceso al trabajador, errores, pérdidas, averías o "
            "adquisición de artículos producidos por la empresa o establecimiento. "
            "La cantidad exigible en ningún caso podrá ser mayor del importe de los "
            "salarios de un mes y el descuento será el que convengan el trabajador y "
            "el patrón, sin que pueda ser mayor del treinta por ciento del excedente "
            "del salario mínimo."
        ),
    },
    {
        "metadata": {"fuente": "LFT", "articulo": "Artículo 56."},
        "text": (
            "### Artículo 56.- Las condiciones de trabajo en ningún caso podrán ser "
            "inferiores a las fijadas en esta Ley y deberán ser proporcionales a la "
            "importancia de los servicios e iguales para trabajos iguales, sin que "
            "puedan establecerse diferencias por motivo de origen étnico o nacionalidad, "
            "género, edad, discapacidad, condición social, condiciones de salud, "
            "religión, condición migratoria, opiniones, preferencias sexuales, estado "
            "civil o cualquier otro que atente contra la dignidad humana."
        ),
    },
]


# ---------------------------------------------------------------------------
# 1. DETECCIÓN DE BACKEND
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """
    Retorna: 'cuda' | 'ollama'
    - cuda  : GPU disponible → carga base model + LoRA adapters
    - ollama: sin GPU → usa modelo exportado 'abogado-virtual' en Ollama
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detectada: {gpu} ({vram:.1f} GB VRAM) → backend CUDA")
            return "cuda"
    except ImportError:
        pass

    # Verificar si Ollama tiene el modelo fine-tuneado exportado
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                print(f"  Ollama detectado — modelo '{OLLAMA_MODEL}' disponible → backend OLLAMA")
                return "ollama"
            else:
                available = ", ".join(models) if models else "ninguno"
                print(f"\n[ERROR] Ollama está corriendo pero no tiene '{OLLAMA_MODEL}'.")
                print(f"  Modelos disponibles: {available}")
                print("\n  Solución: exporta el modelo fine-tuneado a GGUF con:")
                print("    1. En Colab: !python /content/scripts/export_to_gguf.py")
                print("    2. Descarga abogado_virtual_Q4_K_M.gguf y Modelfile a tu PC")
                print("    3. En una carpeta local ejecuta:")
                print("         ollama create abogado-virtual -f Modelfile")
                print("    4. Vuelve a ejecutar este script.")
                sys.exit(1)
    except requests.exceptions.ConnectionError:
        pass

    print("\n[ERROR] No se encontró GPU ni Ollama disponible.")
    print("\n  Opción A — Exportar modelo a Ollama (CPU, recomendado):")
    print("    1. En Colab (GPU): !python /content/scripts/export_to_gguf.py")
    print("    2. Descarga los archivos generados (.gguf + Modelfile)")
    print("    3. Instala Ollama: https://ollama.com/download/windows")
    print("    4. Ejecuta: ollama create abogado-virtual -f Modelfile")
    print()
    print("  Opción B — Ejecutar con GPU en Colab:")
    print("    Sube este script + adaptadores y corre con T4 GPU.")
    sys.exit(1)


def check_prerequisites(backend: str) -> None:
    errors = []

    if CONTRACT_PATH is None:
        errors.append(f"No se encontró '*malas_practicas.md' en {_contracts_dir}")

    if backend == "cuda":
        if not Path(ADAPTER_PATH).exists():
            errors.append(
                f"Adaptadores LoRA no encontrados en: {ADAPTER_PATH}\n"
                "  Descomprime abogado_virtual_lora.zip ahí."
            )

    if errors:
        print("\n[ERROR] No se puede continuar:\n")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. RAG — Recuperación de artículos (con fallback si Docker no está activo)
# ---------------------------------------------------------------------------

def retrieve_relevant_articles(contract_text: str) -> tuple[list[dict], str]:
    """
    Retorna (articles, source) donde source es 'chromadb' o 'fallback'.
    """
    try:
        import chromadb
        from langchain_community.embeddings import HuggingFaceEmbeddings

        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()

        embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"  Cargando embeddings ({embed_model})...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        query_vector = embeddings.embed_query(contract_text[:2000])
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=TOP_K_ARTICLES,
            include=["documents", "metadatas"],
        )
        articles = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return articles, "chromadb"

    except Exception as exc:
        print(f"  ChromaDB no disponible ({exc.__class__.__name__}) — usando artículos de fallback.")
        return FALLBACK_ARTICLES, "fallback"


# ---------------------------------------------------------------------------
# 3. CARGA DEL MODELO BASE + ADAPTADORES LoRA
# ---------------------------------------------------------------------------

def load_finetuned_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Modelo base : {BASE_MODEL_ID}")
    print(f"  Adaptadores : {ADAPTER_PATH}")
    print("  (Primera vez descarga el modelo base ~4.5 GB)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Aplica los adaptadores LoRA sobre el modelo base
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros activos (LoRA): {active:,} / {total:,} ({100*active/total:.3f}%)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL PROMPT
# ---------------------------------------------------------------------------

def build_prompt(contract_text: str, articles: list[dict]) -> str:
    context_blocks = []
    for i, art in enumerate(articles, 1):
        fuente  = art["metadata"].get("fuente", "LEY")
        articulo = art["metadata"].get("articulo", "")
        header  = f"[Artículo {i} — {fuente}] {articulo}".strip()
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

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ---------------------------------------------------------------------------
# 5A. INFERENCIA — Ollama (CPU, modelo exportado)
# ---------------------------------------------------------------------------

def run_ollama_inference(system_prompt: str, user_message: str) -> str:
    print(f"  Modelo Ollama: {OLLAMA_MODEL} (fine-tuneado, CPU)")
    print("  Esto puede tardar 2-5 minutos en CPU — es normal.")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS, "num_ctx": 4096},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=3600)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        sys.exit("[ERROR] Timeout tras 60 min. El modelo es demasiado lento en este CPU.")
    except requests.exceptions.RequestException as exc:
        sys.exit(f"[ERROR] Fallo al comunicarse con Ollama: {exc}")


# ---------------------------------------------------------------------------
# 5B. INFERENCIA — GPU (modelo base + LoRA)
# ---------------------------------------------------------------------------

def run_gpu_inference(model, tokenizer, prompt: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Tokens de entrada: {input_len:,} | Máx. tokens de salida: {MAX_NEW_TOKENS}")

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
    contract_path: Path,
    articles: list[dict],
    rag_source: str,
    analysis: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rag_note = (
        "Artículos recuperados de ChromaDB (Docker local)"
        if rag_source == "chromadb"
        else "Artículos de fallback hardcodeados (ChromaDB no disponible)"
    )

    articles_section = ""
    for i, art in enumerate(articles, 1):
        meta     = art["metadata"]
        fuente   = meta.get("fuente", "LEY")
        articulo = meta.get("articulo", "(sin número)")
        articles_section += f"**{i}. [{fuente}]** {articulo}\n\n"
        articles_section += f"> {art['text'][:300].strip()}...\n\n"

    report = textwrap.dedent(f"""\
        # Análisis Post Fine-Tuning — Detección de Cláusulas Abusivas

        **Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        **Modelo base:** `{BASE_MODEL_ID}`
        **Adaptadores LoRA:** `{ADAPTER_PATH}`
        **Contrato analizado:** `{contract_path.name}`
        **RAG:** {rag_note}

        ---

        ## Artículos Utilizados

        {articles_section}
        ---

        ## Análisis del Modelo Fine-Tuneado (Post LoRA)

        {analysis}

        ---

        ## Cómo Comparar con el Baseline

        Abre en paralelo:
        - `reports/baseline_analysis.md`  — modelo sin entrenar
        - `reports/finetuned_analysis.md` — este reporte (modelo fine-tuneado)

        Señales de mejora esperadas:
        - Cita artículos específicos de la LFT (Art. 5, 87, 80, 110, 56)
        - Identifica las 3 violaciones del contrato de prueba
        - Respuesta más estructurada y precisa legalmente
    """)

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"  Reporte guardado en: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    sep = "=" * 58
    print(sep)
    print("  Eval Fine-Tuneado — Abogado Virtual")
    print(sep)

    print("\n[0/5] Detectando backend...")
    backend = detect_backend()

    print("\n[1/5] Verificando prerequisitos...")
    check_prerequisites(backend)
    print("  OK")

    print(f"\n[2/5] Cargando contrato: {CONTRACT_PATH.name}")
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    print(f"  {len(contract_text):,} caracteres leídos")

    print("\n[3/5] Recuperando artículos de la LFT...")
    articles, rag_source = retrieve_relevant_articles(contract_text)
    for i, art in enumerate(articles, 1):
        meta  = art["metadata"]
        label = f"[{meta.get('fuente','?')}] {meta.get('articulo','')[:60]}"
        print(f"  {i}. {label}")
    print(f"  Fuente RAG: {rag_source}")

    print(f"\n[4/5] Ejecutando inferencia ({backend.upper()})...")
    system_prompt, user_message = (SYSTEM_PROMPT, None)

    if backend == "ollama":
        # Construye user_message sin el template de Llama-3
        context_blocks = []
        for i, art in enumerate(articles, 1):
            fuente   = art["metadata"].get("fuente", "LEY")
            articulo = art["metadata"].get("articulo", "")
            context_blocks.append(
                f"[Artículo {i} — {fuente}] {articulo}\n{art['text'].strip()}"
            )
        contract_excerpt = contract_text[:MAX_CONTRACT_CHARS]
        user_message = (
            f"### ARTÍCULOS RELEVANTES DE LA LFT:\n\n"
            + "\n\n---\n\n".join(context_blocks)
            + f"\n\n---\n\n### CONTRATO:\n\n{contract_excerpt}\n\n---\n\n"
            "Analiza cláusula por cláusula e indica las violaciones encontradas."
        )
        analysis = run_ollama_inference(system_prompt, user_message)
    else:
        prompt   = build_prompt(contract_text, articles)
        model, tokenizer = load_finetuned_model()
        analysis = run_gpu_inference(model, tokenizer, prompt)

    print("\n--- ANÁLISIS GENERADO ---")
    print(analysis)
    print("-------------------------\n")

    print("[5/5] Guardando reporte...")
    save_report(CONTRACT_PATH, articles, rag_source, analysis)

    print(f"\n{sep}")
    print("  EVALUACIÓN FINE-TUNEADO COMPLETA")
    print(f"  Reporte : {OUTPUT_PATH}")
    print(f"  Comparar: reports/baseline_analysis.md")
    print(sep)


if __name__ == "__main__":
    main()
