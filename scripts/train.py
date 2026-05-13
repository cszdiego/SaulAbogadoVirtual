"""
Fine-tuning de LLaMA-3-8B-Instruct con LoRA para detección de cláusulas abusivas.

Soporta dos backends (seleccionado automáticamente):
  1. Unsloth  — 2x más rápido, ~40% menos VRAM. Requiere:
                pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  2. HuggingFace PEFT + TRL  — siempre disponible si hay CUDA.
                pip install trl datasets

⚠️  REQUIERE GPU NVIDIA con CUDA y >= 8 GB VRAM.
    Sin GPU local → usar Google Colab (subir train.jsonl + este script).

Ejecución (desde la raíz del proyecto, venv activo, Docker corriendo):
    python scripts/train.py
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración — todos los hiperparámetros en un solo lugar
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_ID        = "unsloth/llama-3-8b-instruct-bnb-4bit"
DATASET_PATH    = str(BASE_DIR / "data" / "train.jsonl")
OUTPUT_DIR      = str(BASE_DIR / "models" / "abogado_virtual_lora")

MAX_SEQ_LENGTH  = 2048

# LoRA
LORA_R              = 16
LORA_ALPHA          = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT        = 0
LORA_BIAS           = "none"

# Entrenamiento
BATCH_SIZE          = 1
GRAD_ACCUM_STEPS    = 8        # batch efectivo = 1 * 8 = 8
MAX_STEPS           = 60
LEARNING_RATE       = 2e-4
WARMUP_STEPS        = 5
WEIGHT_DECAY        = 0.01
LR_SCHEDULER        = "linear"
LOGGING_STEPS       = 10
SEED                = 42

# Mapa de roles ShareGPT → estándar
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


# ---------------------------------------------------------------------------
# 0. Verificaciones previas
# ---------------------------------------------------------------------------

def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("[ERROR] PyTorch no instalado. Ejecuta: pip install torch")

    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA no detectado.")
        print("\n  Opciones para entrenar sin GPU local:")
        print("  A) Google Colab (gratuito con T4):")
        print("     https://colab.research.google.com/")
        print("     → Sube data/train.jsonl y scripts/train.py")
        print("     → Ejecuta: !python train.py")
        print()
        print("  B) Kaggle Notebooks (gratuito con P100/T4):")
        print("     https://www.kaggle.com/code")
        print()
        print("  C) RunPod / Lambda Labs (pago, ~$0.40/h con A10G):")
        print("     https://www.runpod.io/")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu} ({vram:.1f} GB VRAM)")

    if vram < 7.5:
        print(f"  [ADVERTENCIA] {vram:.1f} GB VRAM puede ser insuficiente.")
        print("  Reduce BATCH_SIZE=1 o usa MODEL_ID con menor número de parámetros.")


def check_dataset() -> None:
    if not Path(DATASET_PATH).exists():
        sys.exit(
            f"[ERROR] Dataset no encontrado: {DATASET_PATH}\n"
            "Ejecuta primero: python scripts/prepare_dataset.py"
        )
    count = sum(1 for _ in open(DATASET_PATH, encoding="utf-8"))
    print(f"  Dataset: {DATASET_PATH} ({count} ejemplos)")


# ---------------------------------------------------------------------------
# 1. Detección de backend
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    try:
        import unsloth  # noqa: F401
        return "unsloth"
    except ImportError:
        pass

    try:
        import trl  # noqa: F401
        return "hf"
    except ImportError:
        sys.exit(
            "[ERROR] Ningún backend de fine-tuning disponible.\n"
            "Instala TRL: pip install trl datasets\n"
            "O Unsloth (más rápido): "
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )


# ---------------------------------------------------------------------------
# 2. Carga del modelo y tokenizador
# ---------------------------------------------------------------------------

def load_model_unsloth():
    from unsloth import FastLanguageModel

    print(f"  Backend: Unsloth | Modelo: {MODEL_ID}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # Unsloth detecta automáticamente float16/bfloat16
        load_in_4bit=True,
    )
    return model, tokenizer


def load_model_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Backend: HuggingFace PEFT + TRL | Modelo: {MODEL_ID}")
    print("  (Primera ejecución descarga ~4.5 GB — solo ocurre una vez)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # evita warnings en entrenamiento

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Necesario antes de aplicar LoRA a un modelo cuantizado
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.enable_input_require_grads()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Aplicar LoRA
# ---------------------------------------------------------------------------

def apply_lora_unsloth(model):
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing="unsloth",   # optimización de memoria de Unsloth
        random_state=SEED,
    )
    return model


def apply_lora_hf(model):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model


def print_trainable_params(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total
    print(f"  Parámetros totales   : {total:,}")
    print(f"  Parámetros LoRA      : {trainable:,} ({pct:.4f}%)")


# ---------------------------------------------------------------------------
# 4. Dataset — carga y formateo con chat template
# ---------------------------------------------------------------------------

def load_and_format_dataset(tokenizer):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"  Ejemplos cargados: {len(dataset)}")

    def apply_chat_template(examples):
        texts = []
        for convs in examples["conversations"]:
            messages = [
                {"role": ROLE_MAP[c["from"]], "content": c["value"]}
                for c in convs
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    print(f"  Tokens en ejemplo 1 (aprox): "
          f"{len(tokenizer.encode(dataset[0]['text'])):,}")
    return dataset


# ---------------------------------------------------------------------------
# 5. Configurar SFTTrainer
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, dataset, backend: str):
    """
    Construye SFTTrainer de forma compatible con cualquier versión de TRL.

    Historial de cambios relevantes en TRL:
      < 0.12  : SFTTrainer acepta tokenizer= y max_seq_length= directamente
      >= 0.12 : tokenizer → processing_class; parámetros SFT van en SFTConfig
      >= 0.15 : max_seq_length eliminado de SFTConfig (el modelo trunca solo)
    """
    import inspect
    import trl as _trl
    from trl import SFTTrainer

    trl_minor = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    new_api = trl_minor >= (0, 12)   # True para TRL 0.12+
    print(f"  TRL {_trl.__version__} — API {'nueva (0.12+)' if new_api else 'clásica (<0.12)'}")

    # Hiperparámetros de entrenamiento base — siempre estables
    base_train_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_8bit",
        fp16=False,
        bf16=True,                       # modelo carga en BF16; fp16+BF16 causa error en gradscaler
        gradient_checkpointing=True,     # libera ~40% VRAM recomputando activaciones
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        report_to="none",
        save_strategy="no",
    )

    if new_api:
        # TRL 0.12+ — usar SFTConfig; tokenizer se pasa como processing_class
        from trl import SFTConfig

        # max_seq_length, dataset_text_field y packing pueden o no existir
        # en SFTConfig según la sub-versión exacta → detectar con inspect
        cfg_params = inspect.signature(SFTConfig.__init__).parameters
        sft_extra = {}
        if "max_seq_length" in cfg_params:
            sft_extra["max_seq_length"] = MAX_SEQ_LENGTH
        if "dataset_text_field" in cfg_params:
            sft_extra["dataset_text_field"] = "text"
        if "packing" in cfg_params:
            sft_extra["packing"] = False

        args = SFTConfig(**base_train_kwargs, **sft_extra)

        return SFTTrainer(
            model=model,
            processing_class=tokenizer,   # nuevo nombre desde TRL 0.12
            train_dataset=dataset,
            args=args,
        )

    else:
        # TRL < 0.12 — API clásica con tokenizer= y kwargs directos en SFTTrainer
        from transformers import TrainingArguments

        args = TrainingArguments(**base_train_kwargs)

        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs: dict = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": dataset,
            "args": args,
        }
        if "max_seq_length" in sft_params:
            trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        if "dataset_text_field" in sft_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "packing" in sft_params:
            trainer_kwargs["packing"] = False

        return SFTTrainer(**trainer_kwargs)


# ---------------------------------------------------------------------------
# 6. Guardar adaptadores LoRA y tokenizador
# ---------------------------------------------------------------------------

def save_artifacts(model, tokenizer, backend: str) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if backend == "unsloth":
        # Guarda solo los pesos LoRA (ligeros, ~15 MB)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    else:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    # Verifica que los archivos se crearon
    saved = list(Path(OUTPUT_DIR).glob("*"))
    print(f"  Archivos guardados ({len(saved)}):")
    for f in sorted(saved):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:8.1f} KB")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    sep = "=" * 58
    print(sep)
    print("  Fine-Tuning LoRA — Abogado Virtual")
    print(sep)

    print("\n[0/6] Verificando GPU y dataset...")
    check_cuda()
    check_dataset()

    print("\n[1/6] Detectando backend...")
    backend = detect_backend()
    print(f"  Backend: {backend.upper()}")

    print(f"\n[2/6] Cargando modelo base ({MODEL_ID})...")
    if backend == "unsloth":
        model, tokenizer = load_model_unsloth()
    else:
        model, tokenizer = load_model_hf()

    print(f"\n[3/6] Aplicando LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    if backend == "unsloth":
        model = apply_lora_unsloth(model)
    else:
        model = apply_lora_hf(model)
    print_trainable_params(model)

    print("\n[4/6] Cargando y formateando dataset...")
    dataset = load_and_format_dataset(tokenizer)

    print("\n[5/6] Iniciando entrenamiento...")
    effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"  Batch efectivo : {effective_batch} | Pasos: {MAX_STEPS} | LR: {LEARNING_RATE}")
    print(f"  Épocas aprox.  : {(MAX_STEPS * effective_batch) / len(dataset):.1f}")

    trainer = build_trainer(model, tokenizer, dataset, backend)
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics.get("train_runtime", 0) / 60
    loss = trainer_stats.metrics.get("train_loss", float("nan"))
    print(f"\n  Entrenamiento completado en {runtime_min:.1f} min | Loss final: {loss:.4f}")

    print(f"\n[6/6] Guardando adaptadores LoRA en: {OUTPUT_DIR}")
    save_artifacts(model, tokenizer, backend)

    print(f"\n{sep}")
    print("  FINE-TUNING COMPLETO")
    print(f"  Adaptadores: {OUTPUT_DIR}")
    print(sep)
    print()
    print("  Próximo paso — inferencia con el modelo fine-tuneado:")
    print("  from peft import PeftModel")
    print("  from transformers import AutoModelForCausalLM")
    print(f'  model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
