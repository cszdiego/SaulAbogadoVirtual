"""
Fusiona los adaptadores LoRA con el modelo base y exporta a GGUF para Ollama.

Ejecutar en Google Colab (GPU T4) después de hacer el fine-tuning:
    !python export_to_gguf.py

Resultado: descarga abogado_virtual_Q4_K_M.gguf (~4 GB) a tu PC.
Luego importa en Ollama local con:
    ollama create abogado-virtual -f Modelfile
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Detectar si estamos en Colab o local
# ---------------------------------------------------------------------------
IN_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

if IN_COLAB:
    BASE_DIR     = Path("/content")
    ADAPTER_PATH = BASE_DIR / "models" / "abogado_virtual_lora"
    MERGED_DIR   = BASE_DIR / "merged_model"
    GGUF_PATH    = BASE_DIR / "abogado_virtual_Q4_K_M.gguf"
    MODELFILE    = BASE_DIR / "Modelfile"
else:
    BASE_DIR     = Path(__file__).resolve().parent.parent
    ADAPTER_PATH = BASE_DIR / "models" / "abogado_virtual_lora"
    MERGED_DIR   = BASE_DIR / "models" / "merged_model"
    GGUF_PATH    = BASE_DIR / "models" / "abogado_virtual_Q4_K_M.gguf"
    MODELFILE    = BASE_DIR / "models" / "Modelfile"

BASE_MODEL_ID = "unsloth/llama-3-8b-instruct-bnb-4bit"

SYSTEM_PROMPT = (
    "Eres un experto legal mexicano especializado en derecho laboral. "
    "Tu tarea es analizar contratos individuales de trabajo y detectar "
    "violaciones a la Ley Federal del Trabajo (LFT). Para cada irregularidad "
    "cita: (1) la cláusula del contrato, (2) el artículo de la LFT violado, "
    "y (3) una explicación precisa. Si el contrato cumple con la normativa, indícalo."
)


# ---------------------------------------------------------------------------
# Paso 1: Fusionar LoRA con el modelo base
# ---------------------------------------------------------------------------

def merge_adapters() -> None:
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not ADAPTER_PATH.exists():
        sys.exit(f"[ERROR] Adaptadores no encontrados en: {ADAPTER_PATH}")

    print(f"  Cargando modelo base: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Aplicando adaptadores LoRA desde: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))

    print("  Fusionando pesos (merge_and_unload)...")
    merged = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Guarda config sin quantization_config (el modelo fusionado ya no es cuantizado)
    print("  Guardando config...")
    config = merged.config
    config.quantization_config = None
    config.save_pretrained(str(MERGED_DIR))

    # Libera VRAM antes de construir el state dict en RAM
    import gc
    print("  Liberando VRAM...")
    merged = merged.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Guarda pesos manualmente en FP16 para evitar el bug de
    # revert_weight_conversion en transformers >= 4.47
    # Este paso tarda ~5-10 minutos — no interrumpir
    all_params = list(merged.named_parameters())
    total = len(all_params)
    print(f"  Guardando {total} tensores en FP16 (tarda ~5-10 min, no interrumpir)...")

    state_dict = {}
    for i, (name, param) in enumerate(all_params, 1):
        tensor = param.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
        state_dict[name] = tensor
        if i % 30 == 0 or i == total:
            print(f"\r  Progreso: {i}/{total} tensores", end="", flush=True)

    print()
    save_file(state_dict, str(MERGED_DIR / "model.safetensors"))
    del state_dict
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"  Modelo fusionado guardado en: {MERGED_DIR}")


# ---------------------------------------------------------------------------
# Paso 2: Convertir a GGUF con llama.cpp
# ---------------------------------------------------------------------------

def convert_to_gguf() -> None:
    import subprocess

    llama_cpp_dir = Path("/content/llama.cpp") if IN_COLAB else Path("llama.cpp")

    if not llama_cpp_dir.exists():
        print("  Clonando llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ggerganov/llama.cpp",
             str(llama_cpp_dir)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "-r", str(llama_cpp_dir / "requirements.txt"), "-q"],
            check=True,
        )

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Versiones más antiguas usan convert.py
        convert_script = llama_cpp_dir / "convert.py"

    print(f"  Convirtiendo a GGUF Q4_K_M...")
    print(f"  Salida: {GGUF_PATH}")

    subprocess.run(
        [
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(GGUF_PATH),
            "--outtype", "q4_k_m",
        ],
        check=True,
    )
    size_gb = GGUF_PATH.stat().st_size / 1e9
    print(f"  GGUF generado: {size_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Paso 3: Crear Modelfile para Ollama
# ---------------------------------------------------------------------------

def create_modelfile() -> None:
    content = f"""FROM ./abogado_virtual_Q4_K_M.gguf

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0
PARAMETER num_predict 1024
PARAMETER num_ctx 4096
"""
    MODELFILE.write_text(content, encoding="utf-8")
    print(f"  Modelfile creado en: {MODELFILE}")


# ---------------------------------------------------------------------------
# Paso 4: Descargar desde Colab
# ---------------------------------------------------------------------------

def download_from_colab() -> None:
    if not IN_COLAB:
        return

    try:
        from google.colab import files
        print("\n  Descargando archivos a tu PC...")
        print("  ⚠️  El GGUF pesa ~4 GB — puede tardar varios minutos")
        files.download(str(GGUF_PATH))
        files.download(str(MODELFILE))
    except Exception as exc:
        print(f"  Descarga automática falló ({exc}).")
        print(f"  Descarga manual: panel izquierdo de Colab → {GGUF_PATH}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 56
    print(sep)
    print("  Exportar Modelo Fine-Tuneado → GGUF para Ollama")
    print(sep)

    print("\n[1/4] Fusionando adaptadores LoRA con el modelo base...")
    merge_adapters()

    print("\n[2/4] Convirtiendo a GGUF Q4_K_M...")
    convert_to_gguf()

    print("\n[3/4] Creando Modelfile para Ollama...")
    create_modelfile()

    print("\n[4/4] Descargando archivos...")
    download_from_colab()

    print(f"\n{sep}")
    print("  EXPORTACIÓN COMPLETA")
    print(sep)
    print()
    print("  Pasos siguientes en tu PC:")
    print("  1. Mueve abogado_virtual_Q4_K_M.gguf y Modelfile a una carpeta")
    print("  2. Desde esa carpeta ejecuta:")
    print("       ollama create abogado-virtual -f Modelfile")
    print("  3. Verifica que el modelo se creó:")
    print("       ollama list")
    print("  4. Corre la evaluación:")
    print("       python scripts/eval_finetuned.py")


if __name__ == "__main__":
    main()
